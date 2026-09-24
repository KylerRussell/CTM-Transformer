"""Read-only attention capture for the frozen research model presets."""
from collections import defaultdict
import torch
import torch.nn.functional as F


class AttentionCapture:
    """Retain native outputs; capture CTM probabilities or recompute SDPA's last row."""
    def __init__(self,model,family):
        self.model,self.family=model,family
        self.handles=[];self.calls=defaultdict(int);self.records=[];self.label=None
        self.max_reconstruction_error=0.0

    def begin(self):
        self.calls.clear();self.records=[];self.label=None

    def _name(self,name):
        self.calls[name]+=1
        return f'{name}/call{self.calls[name]}'

    def __enter__(self):
        if self.model.training:raise ValueError('Capture requires eval mode')
        if self.family=='ctm':
            for name,module in self.model.named_modules():
                if name.endswith('.attn_dropout'):
                    def hook(module,args,output,name=name):
                        self.records.append((self._name(name),output[:,:,-1,:].detach().float().cpu()))
                    self.handles.append(module.register_forward_hook(hook))
        else:
            for name,module in self.model.named_modules():
                if name.endswith('.attn'):
                    def pre(module,args,name=name):self.label=self._name(name)
                    self.handles.append(module.register_forward_pre_hook(pre))
            self.native=F.scaled_dot_product_attention
            def wrapped(q,k,v,*args,**kwargs):
                if args or kwargs.get('attn_mask') is not None or kwargs.get('dropout_p',0)!=0 or not kwargs.get('is_causal') or q.shape[-2]!=k.shape[-2]:
                    raise ValueError('Only the unpadded square causal diagnostic path is supported')
                output=self.native(q,k,v,**kwargs)
                with torch.autocast('cuda',enabled=False):
                    scale=kwargs.get('scale') or q.shape[-1]**-.5
                    probs=(q[:,:,-1:,:].float() @ k.float().transpose(-2,-1)*scale).softmax(-1)
                    reconstructed=probs @ v.float()
                    reference=output[:,:,-1:,:].float()
                    error=((reconstructed-reference).abs()/(1+reference.abs())).max().item()
                    self.max_reconstruction_error=max(self.max_reconstruction_error,error)
                    if error>.02:raise ValueError(f'SDPA reconstruction discrepancy {error}')
                self.records.append((self.label,probs.squeeze(-2).detach().cpu()))
                return output
            F.scaled_dot_product_attention=wrapped
        return self

    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        if hasattr(self,'native'):F.scaled_dot_product_attention=self.native


def prompt_positions(row):
    """Return BOS-adjusted positions of source labels, values, and query label."""
    sources=[1+row['prompt'].index(n+'>') for n in 'ABCDEFGH']
    values=[p+2 for p in sources]
    query=1+row['prompt'].index(';start ')+len(';start ')
    assert row['prompt'][query-1]==row['start']
    return sources,values,query

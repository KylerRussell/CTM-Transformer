import os
import pytest
import torch
from ctm_transformer.attention_diagnostic import AttentionCapture,prompt_positions
from ctm_transformer.algorithmic import AlgorithmicTokenizer
from scripts.eval_harness import _load_checkpoint

@pytest.mark.parametrize('family,events',[('transformer',2),('recurrent_depth',10),('ctm',8)])
def test_capture_preserves_native_logits_and_cleans_up(family,events):
    device=os.environ.get('CTM_TEST_DEVICE','cuda:0')
    model,_=_load_checkpoint(f'research/runs/ordered_long_v1/seed17/{family}/best.pt')
    model.to(device).eval();torch.set_num_threads(4)
    import json
    from pathlib import Path
    row=json.loads(Path('research/data/query_diagnostic_v1/pointer/validation_start_A.jsonl').read_text().splitlines()[0])
    tok=AlgorithmicTokenizer();ids=torch.tensor([[tok.bos_token]+tok.encode(row['prompt'])],device=device)
    native=torch.nn.functional.scaled_dot_product_attention
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        before=model(ids)['logits']
        with AttentionCapture(model,family) as cap:
            cap.begin();after=model(ids)['logits']
            assert len(cap.records)==events
            for _,p in cap.records:torch.testing.assert_close(p.sum(-1),torch.ones_like(p.sum(-1)),atol=.005,rtol=0)
        assert torch.equal(before,after)
    assert torch.nn.functional.scaled_dot_product_attention is native
    sources,values,query=prompt_positions(row)
    assert row['prompt'][values[0]-1]==row['answer']
    assert row['prompt'][sources[0]-1]=='A' and row['prompt'][query-1]=='A'

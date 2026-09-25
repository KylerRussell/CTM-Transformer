"""Dense-supervision pointer task: several queries answered per map.

Each sequence presents one single 12-cycle map in random edge order and asks
for the k-hop successor of QUERIES distinct nodes in random order:

    map H>J;L>K;...;hops 2;ask KCA...=JBE...<EOS>

All answers and EOS are supervised, giving QUERIES answers per map instead of
one. Answers are scored teacher-forced, per answer token. Because a k-hop
successor map is a permutation, earlier gold answers exclude candidates for
later ones; with 6 of 12 nodes queried, a guesser that exploits this exclusion
averages at most 12.3% per answer (chance 1/11 = 9.09%). Accuracy by answer
position is reported so that any such use is visible.

This module is independent of the frozen single-query `pointer` format.
"""
from collections import Counter,defaultdict
import hashlib,json,random,string
from pathlib import Path
import torch
from torch.nn import functional as F
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,content_hash,validate_record

TASK='pointer_dense';NODES=12;QUERIES=6


def labels(nodes=NODES):return list(string.ascii_uppercase[:nodes])


def hop(successors,node,steps):
    for _ in range(steps):node=successors[node]
    return node


def dense_prompt(record):
    edges=';'.join(f"{n}>{record['successors'][n]}" for n in record['presentation'])
    return f"map {edges};hops {record['steps']};ask {''.join(record['queries'])}="


def map_key(record):return content_hash(['map',record['successors']])


def dense_identity(record):
    # Maps, not map-query pairs, are held out across splits, as for the single-query format.
    return content_hash([TASK,record['successors']])


def dense_record(rng,steps,nodes=NODES,queries=QUERIES):
    names=labels(nodes);cycle=rng.sample(names,nodes)
    successors={cycle[i]:cycle[(i+1)%nodes] for i in range(nodes)}
    presentation=rng.sample(names,nodes)
    while presentation==names:presentation=rng.sample(names,nodes)
    record={'task':TASK,'successors':successors,'presentation':presentation,'steps':steps,
            'queries':rng.sample(names,queries),'difficulty':{'nodes':nodes,'steps':steps}}
    record['prompt']=dense_prompt(record)
    record['answer']=''.join(hop(successors,q,steps) for q in record['queries'])
    record['id']=dense_identity(record)
    return record


def validate_dense_record(record):
    if record.get('task')!=TASK:raise ValueError('Not a dense pointer record')
    mapping=record['successors'];nodes=len(mapping);names=labels(nodes)
    if not 2<=nodes<=26 or set(mapping)!=set(names) or set(mapping.values())!=set(names):raise ValueError('Invalid node permutation')
    seen,node=set(),names[0]
    while node not in seen:seen.add(node);node=mapping[node]
    if len(seen)!=nodes:raise ValueError('Pointer mapping must be one cycle')
    if sorted(record['presentation'])!=names or record['presentation']==names:raise ValueError('Invalid or canonical edge presentation')
    q=record['queries']
    if not q or len(set(q))!=len(q) or not set(q)<=set(names):raise ValueError('Queries must be distinct nodes')
    if type(record['steps']) is not int or not 1<=record['steps']<nodes:raise ValueError('Invalid hop count')
    if (record.get('prompt')!=dense_prompt(record) or record.get('answer')!=''.join(hop(mapping,x,record['steps']) for x in q)
            or record.get('id')!=dense_identity(record) or record.get('difficulty')!={'nodes':nodes,'steps':record['steps']}):
        raise ValueError('Record prompt, solution, identity, or difficulty is inconsistent')


class DensePointerDataset:
    """Same batch interface as `AnswerDataset`; supervises every answer token and EOS."""
    def __init__(self,path,tokenizer,seq_len):
        self.tokenizer=tokenizer;raw=Path(path).read_bytes()
        self.records=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        if not self.records:raise ValueError('Empty dense dataset')
        self.ids=set();xs,ys,lengths=[],[],[]
        for record in self.records:
            validate_dense_record(record)
            if record['id'] in self.ids:raise ValueError('Duplicate semantic instance')
            self.ids.add(record['id'])
            prefix=[tokenizer.bos_token]+tokenizer.encode(record['prompt'])
            answer=tokenizer.encode(record['answer'])+[tokenizer.eot_token]
            combined=prefix+answer
            if len(combined)-1>seq_len:raise ValueError('Example exceeds seq_len; answer truncation is not allowed')
            x,y=combined[:-1],[-100]*(len(prefix)-1)+answer
            lengths.append(len(x));xs.append(x+[tokenizer.pad_token]*(seq_len-len(x)));ys.append(y+[-100]*(seq_len-len(y)))
        self.inputs,self.targets,self.lengths=torch.tensor(xs),torch.tensor(ys),torch.tensor(lengths)
        self.metadata={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'examples':len(self.records),
            'max_input_length':max(lengths),'supervised_tokens':int((self.targets!=-100).sum()),'task':TASK,
            'queries_per_example':len(self.records[0]['queries']),'splits':sorted({r.get('split') for r in self.records})}

    def __len__(self):return len(self.records)

    def batch(self,index):
        lengths=self.lengths[index];width=int(lengths.max())
        return self.inputs[index,:width],self.targets[index,:width],int(lengths.sum())


def prior_map_ids(data_root,output_root,nodes=NODES):
    """Format-independent keys of every earlier 12-node map, single-query or dense."""
    output_root=Path(output_root).resolve();seen=set();files={}
    for path in sorted(Path(data_root).rglob('*.jsonl')):
        if path.resolve().is_relative_to(output_root):continue
        raw=path.read_bytes();count=0
        for line in raw.decode().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get('task') in ('pointer',TASK) and len(row.get('successors',{}))==nodes:
                validate_record(row) if row['task']=='pointer' else validate_dense_record(row)
                # Identity hashes carry a task tag, so hold out the map itself across formats.
                seen.add(map_key(row));count+=1
        if count:files[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'maps':count}
    return seen,files


def create_dense_suite(output_root,design,name,seed,data_root='research/data'):
    root=Path(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty dataset directory; generated splits are immutable')
    seen,files=prior_map_ids(data_root,root);excluded=len(seen)
    directory=root/'pointer';directory.mkdir(parents=True)
    manifest={'schema_version':1,'name':name,'task':TASK,'seed':seed,'nodes':NODES,'queries_per_example':QUERIES,
        'design':{s:[n,h] for s,(n,h) in design.items()},'role':'development only; no locked test split',
        'tokenizer':AlgorithmicTokenizer().snapshot(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_data_root':str(data_root),'prior_files':files,'excluded_prior_maps':excluded,
        'split_policy':'Every map appears in exactly one split and in no earlier 12-node data of either format. Presentations are random and noncanonical; queries are distinct random nodes. Hop counts are balanced and interleaved before shuffling.',
        'tasks':{TASK:{}}}
    for index,(split,(per_hop,hops)) in enumerate(design.items()):
        rng=random.Random(seed+index);rows=[]
        for i in range(per_hop*len(hops)):
            while True:
                row=dense_record(rng,hops[i%len(hops)]);key=map_key(row)
                if key not in seen:break
            seen.add(key);row['split']=split;rows.append(row)
        rng.shuffle(rows)
        raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path=directory/(split+'.jsonl');path.write_bytes(raw)
        manifest['tasks'][TASK][split]={'path':str(path.relative_to(root)),'examples':len(rows),'sha256':hashlib.sha256(raw).hexdigest(),
            'steps':dict(sorted(Counter(r['steps'] for r in rows).items()))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_dense_suite(root)
    return manifest


def load_dense_split(root,split,seq_len=128):
    root=Path(root);entry=json.loads((root/'manifest.json').read_text())['tasks'][TASK][split]
    data=DensePointerDataset(root/entry['path'],AlgorithmicTokenizer(),seq_len)
    if data.metadata['sha256']!=entry['sha256'] or len(data)!=entry['examples'] or data.metadata['splits']!=[split]:
        raise ValueError('Dataset does not match its manifest')
    return data


def validate_dense_suite(root,check_prior=True):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());design=m['design']
    data={split:load_dense_split(root,split) for split in m['tasks'][TASK]}
    if set(data)!=set(design):raise ValueError('Splits differ from the declared design')
    maps={}
    for split,d in data.items():
        per_hop,hops=design[split]
        if Counter(r['steps'] for r in d.records)!=Counter({h:per_hop for h in hops}):raise ValueError(f'{split}: hop counts differ from the design')
        if any(len(r['queries'])!=m['queries_per_example'] or len(r['successors'])!=NODES for r in d.records):raise ValueError(f'{split}: wrong shape')
        maps[split]={map_key(r) for r in d.records}
    names=list(maps)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if maps[a]&maps[b]:raise ValueError(f'Map overlap between {a} and {b}')
    if check_prior:
        prior,files=prior_map_ids(m['prior_data_root'],root)
        if files!=m['prior_files'] or len(prior)!=m['excluded_prior_maps']:raise ValueError('Prior inventory changed')
        if any(s&prior for s in maps.values()):raise ValueError('Overlap with prior data')
    return data


def exclusion_guess_ceiling(nodes=NODES,queries=QUERIES):
    """Mean per-answer accuracy of a random guesser that excludes the query node and earlier gold answers."""
    return sum(1/(nodes-1-i) for i in range(queries))/queries


@torch.inference_mode()
def evaluate_dense(model,dataset,config,device,policies=('final','confidence')):
    """Teacher-forced per-answer accuracy, CE and tick diagnostics for final/confidence readouts."""
    from ctm_transformer.research import model_family
    if not policies or any(p not in ('final','confidence') for p in policies):raise ValueError('Unknown or empty readout policy')
    was_training=model.training;model.eval();eot=dataset.tokenizer.eot_token
    totals={p:0.0 for p in policies};count=0;rows={p:[] for p in policies};raw_ce=raw_correct=hist=None
    try:
        for offset in range(0,len(dataset),config.batch_size):
            inputs,targets,_=dataset.batch(slice(offset,offset+config.batch_size));inputs,targets=inputs.to(device),targets.to(device)
            kw={'return_all_logits':True} if model_family(config)!='ctm' else {}
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config.dtype=='bfloat16'):
                output=model(inputs,max_thought_steps=config.max_thought_steps,**kw)
            ticks=torch.stack(output['all_logits'],dim=2)  # B,S,T,V
            mask=targets.ne(-100);gold=targets[mask];logits=ticks[mask].float()
            logp=logits.log_softmax(-1)
            ce=-logp.gather(-1,gold[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
            chosen=(-(logp.exp()*logp).sum(-1)).argmin(-1);index=torch.arange(len(gold),device=gold.device)
            correct=logits.argmax(-1).eq(gold[:,None])
            values={'final':(ce[:,-1],correct[:,-1]),'confidence':(ce[index,chosen],correct[index,chosen])}
            per_row=mask.sum(1).tolist();is_answer=gold.ne(eot)
            for p in policies:
                totals[p]+=values[p][0].sum().item()
                flags=values[p][1][is_answer].tolist();start=0
                for record,n in zip(dataset.records[offset:offset+config.batch_size],per_row):
                    k=n-1;rows[p].append({'id':record['id'],'steps':record['steps'],'correct':flags[start:start+k]});start+=k
            additions=(ce.sum(0),correct.sum(0),torch.bincount(chosen,minlength=logits.shape[1]))
            if raw_ce is None:raw_ce,raw_correct,hist=additions
            else:raw_ce+=additions[0];raw_correct+=additions[1];hist+=additions[2]
            count+=len(gold)
        result={}
        for p in policies:
            by_hop=defaultdict(list);by_position=defaultdict(list)
            for r in rows[p]:
                by_hop[r['steps']].append(r['correct'])
                for i,c in enumerate(r['correct']):by_position[i].append(c)
            summarize=lambda seqs:{'examples':len(seqs),'answer_accuracy':sum(map(sum,seqs))/sum(map(len,seqs)),
                                   'sequence_exact':sum(all(s) for s in seqs)/len(seqs)}
            result[p]={'loss':totals[p]/count,'target_tokens':count,'thought_steps':config.max_thought_steps,
                **summarize([r['correct'] for r in rows[p]]),
                'by_hop':{str(h):summarize(v) for h,v in sorted(by_hop.items())},
                'answer_accuracy_by_position':[sum(v)/len(v) for _,v in sorted(by_position.items())],
                'predictions':rows[p]}
        result['diagnostics']={'raw_tick_ce':(raw_ce/count).tolist(),'raw_tick_token_accuracy':(raw_correct/count).tolist(),
            'confidence_tick_histogram':hist.tolist(),'exclusion_guess_ceiling':exclusion_guess_ceiling(),
            'note':'Teacher-forced over supervised answer and EOS tokens; accuracy fields cover answer tokens only.'}
        return result
    finally:
        model.train(was_training)

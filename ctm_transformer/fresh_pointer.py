"""Fresh one-hop confirmation maps excluded from all inventoried prior data."""
from collections import Counter
import hashlib,itertools,json,random
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,instance_identity,validate_record,load_algorithmic_split
from ctm_transformer.pointer_diagnostics import revise_query


def prior_inventory(data_root,output_root):
    output_root=Path(output_root).resolve();seen=set();files={}
    for path in sorted(Path(data_root).rglob('*.jsonl')):
        if path.resolve().is_relative_to(output_root):continue
        raw=path.read_bytes();files[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
        for line in raw.decode().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get('task')=='pointer' and len(row.get('successors',{}))==8:
                validate_record(row);seen.add(instance_identity(row))
    return seen,files


def create_fresh_suite(output_root,data_root='research/data',seed=20260924,count=512):
    root=Path(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Fresh output directory required')
    if type(count)is not int or count<1:raise ValueError('Positive map count required')
    seen,files=prior_inventory(data_root,root);candidates=[]
    labels=list('ABCDEFGH')
    # Fix A first to enumerate every directed Hamiltonian cycle exactly once.
    for tail in itertools.permutations(labels[1:]):
        cycle=['A',*tail];mapping={cycle[i]:cycle[(i+1)%8] for i in range(8)}
        base={'task':'pointer','successors':mapping,'start':'A','steps':1,'presentation':labels}
        if instance_identity(base) not in seen:candidates.append(base)
    if len(candidates)<count:raise ValueError(f'Only {len(candidates)} unseen maps remain, need {count}')
    rng=random.Random(seed);chosen=rng.sample(candidates,count);ordered=[];shuffled=[]
    for row in chosen:
        start=rng.choice(labels);order=rng.sample(labels,8)
        while order==labels:order=rng.sample(labels,8)
        ordered.append(revise_query(row,start=start,presentation=labels,split='test_id'))
        shuffled.append(revise_query(row,start=start,presentation=order,split='test_shuffled'))
    manifest={'schema_version':1,'name':'fresh_pointer_confirmation_v1','seed':seed,'role':'Locked confirmation only; no training or recipe selection',
        'tokenizer':AlgorithmicTokenizer().snapshot(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_data_root':str(data_root),'prior_files':files,'excluded_map_ids':sorted(seen),'excluded_maps':len(seen),
        'universe_maps':5040,'available_unseen_maps':len(candidates),
        'split_policy':'Ordered and shuffled splits pair the same fresh maps, starts, labels and lengths; maps excluded from every inventoried earlier JSONL.',
        'tasks':{'pointer':{}}}
    directory=root/'pointer';directory.mkdir(parents=True)
    for split,rows in [('test_id',ordered),('test_shuffled',shuffled)]:
        raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path=directory/(split+'.jsonl');path.write_bytes(raw)
        manifest['tasks']['pointer'][split]={'path':str(path.relative_to(root)),'examples':len(rows),
            'sha256':hashlib.sha256(raw).hexdigest(),'answer_counts':dict(Counter(r['answer'] for r in rows))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_fresh_suite(root)
    return manifest


def validate_fresh_suite(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    for path,info in m['prior_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Prior inventory changed')
    prior,files=prior_inventory(m['prior_data_root'],root)
    if files!=m['prior_files'] or prior!=set(m['excluded_map_ids']):raise ValueError('Prior file/map inventory changed')
    a=load_algorithmic_split(root,'pointer','test_id',128);b=load_algorithmic_split(root,'pointer','test_shuffled',128)
    if a.ids&prior or a.ids!=b.ids:raise ValueError('Map overlap or missing pairs')
    for x,y in zip(a.records,b.records):
        if any(x[k]!=y[k] for k in ('id','successors','start','steps','answer')):raise ValueError('Pair semantics changed')
        if x['presentation']!=list('ABCDEFGH') or y['presentation']==x['presentation']:raise ValueError('Invalid presentation pair')
    return m

"""Fresh-map (effectively online) 12-node pointer data with a repeated-map control.

Every map is a single 12-cycle; there are 11! (about 39.9M) such maps, so a
96,000-example training file can contain only unique maps. Each training map
is then seen once, the practical equivalent of online generation with the
frozen file-based trainer. The repeated control is the first 2,048 records of
the fresh one-hop file, so it trains on a nested subset of the same maps.
"""
from collections import Counter
import hashlib,json,random
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,instance_identity,pointer_record,validate_record,load_algorithmic_split

NODES=12
# split: (examples per hop, hops). Training hop counts are balanced by cycling.
SPLITS={
    'train_fresh_onehop':(96000,[1]),
    'train_fresh_multihop':(32000,[1,2,3]),
    'validation':(128,[1,2,3]),
    'eval_id':(256,[1,2,3]),
    'eval_depth':(256,[4,5,6,7,8]),
}
REPEATED=('train_repeated_onehop',2048)
TRAINING=('train_fresh_onehop','train_repeated_onehop','train_fresh_multihop')


def prior_inventory(data_root,output_root,nodes=NODES):
    output_root=Path(output_root).resolve();seen=set();files={}
    for path in sorted(Path(data_root).rglob('*.jsonl')):
        if path.resolve().is_relative_to(output_root):continue
        raw=path.read_bytes();count=0
        for line in raw.decode().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get('task')=='pointer' and len(row.get('successors',{}))==nodes:
                validate_record(row);seen.add(instance_identity(row));count+=1
        if count:files[str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'maps':count}
    return seen,files


def create_online_pointer_suite(output_root,data_root='research/data',seed=20260925):
    root=Path(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty dataset directory; generated splits are immutable')
    seen,files=prior_inventory(data_root,root);excluded=len(seen);rows={}
    canonical=[chr(65+i) for i in range(NODES)]
    for index,(split,(per_hop,hops)) in enumerate(SPLITS.items()):
        rng=random.Random(seed+index);out=[]
        for i in range(per_hop*len(hops)):
            while True:
                row=pointer_record(rng,NODES,hops[i%len(hops)])
                if row['id'] not in seen and row['presentation']!=canonical:break
            seen.add(row['id']);row['split']=split;out.append(row)
        rng.shuffle(out);rows[split]=out
    name,count=REPEATED
    rows[name]=[{**r,'split':name} for r in rows['train_fresh_onehop'][:count]]
    directory=root/'pointer';directory.mkdir(parents=True)
    manifest={'schema_version':1,'name':'online_pointer_v1','seed':seed,'nodes':NODES,
        'role':'development only; no locked test split','tokenizer':AlgorithmicTokenizer().snapshot(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_data_root':str(data_root),'prior_files':files,'excluded_prior_maps':excluded,
        'split_policy':'Every map appears in one split only, except train_repeated_onehop, which is the first 2,048 records of train_fresh_onehop. '
            'Presentations are random per example and never canonical. Hop counts are balanced within each split.',
        'tasks':{'pointer':{}}}
    for split,out in rows.items():
        raw=''.join(canonical_json(r)+'\n' for r in out).encode();path=directory/(split+'.jsonl');path.write_bytes(raw)
        manifest['tasks']['pointer'][split]={'path':str(path.relative_to(root)),'examples':len(out),
            'sha256':hashlib.sha256(raw).hexdigest(),'steps':dict(sorted(Counter(r['steps'] for r in out).items()))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_online_pointer_suite(root)
    return manifest


def validate_online_pointer_suite(root,check_prior=True):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    data={split:load_algorithmic_split(root,'pointer',split,128) for split in m['tasks']['pointer']}
    expected={s:n*len(h) for s,(n,h) in SPLITS.items()}|{REPEATED[0]:REPEATED[1]}
    if {s:len(d) for s,d in data.items()}!=expected:raise ValueError('Split sizes differ from the declared design')
    for split,d in data.items():
        if any(len(r['successors'])!=NODES or r['presentation']==sorted(r['presentation']) for r in d.records):
            raise ValueError(f'{split}: wrong node count or canonical presentation')
        steps=Counter(r['steps'] for r in d.records);hops=SPLITS.get(split,(None,[1]))[1]
        if set(steps)!=set(hops) or len(set(steps.values()))!=1:raise ValueError(f'{split}: unbalanced or undeclared hop counts')
    repeated=data[REPEATED[0]].records;fresh=data['train_fresh_onehop'].records
    if [(r['id'],r['start'],r['presentation']) for r in repeated]!=[(r['id'],r['start'],r['presentation']) for r in fresh[:REPEATED[1]]]:
        raise ValueError('Repeated control is not the declared nested subset')
    independent=[s for s in data if s!=REPEATED[0]]
    for i,a in enumerate(independent):
        for b in independent[i+1:]:
            if data[a].ids&data[b].ids:raise ValueError(f'Map overlap between {a} and {b}')
    if check_prior:
        prior,files=prior_inventory(m['prior_data_root'],root)
        if files!=m['prior_files'] or len(prior)!=m['excluded_prior_maps']:raise ValueError('Prior inventory changed')
        if any(d.ids&prior for d in data.values()):raise ValueError('Overlap with prior data')
    return data

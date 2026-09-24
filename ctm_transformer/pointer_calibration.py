"""Fresh-map 12-node pointer suites for optimizer/budget and difficulty calibration.

Designs are explicit: {split: (examples per hop, hops)}. Every map appears in
exactly one split and is excluded from all earlier 12-node data, so a
training file sized to (updates x batch) presents each map exactly once.
"""
from collections import Counter
import hashlib,json,random
from pathlib import Path
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,pointer_record,load_algorithmic_split
from ctm_transformer.online_pointer import prior_inventory

NODES=12
CALIBRATION_DESIGN={
    'train_fresh_onehop':(960000,[1]),
    'train_fresh_multihop':(240000,[1,2,3,4]),
    'validation':(96,[1,2,3,4]),
    'eval_id':(256,[1,2,3,4]),
    'eval_depth':(256,[5,6,7,8]),
}


def create_pointer_suite(output_root,design,name,seed,data_root='research/data'):
    root=Path(output_root)
    if root.exists() and any(root.iterdir()):raise ValueError('Use an empty dataset directory; generated splits are immutable')
    seen,files=prior_inventory(data_root,root,NODES);excluded=len(seen)
    canonical=[chr(65+i) for i in range(NODES)];directory=root/'pointer';directory.mkdir(parents=True)
    manifest={'schema_version':1,'name':name,'seed':seed,'nodes':NODES,'design':{s:[n,h] for s,(n,h) in design.items()},
        'role':'development only; no locked test split','tokenizer':AlgorithmicTokenizer().snapshot(),
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prior_data_root':str(data_root),'prior_files':files,'excluded_prior_maps':excluded,
        'split_policy':'Every map appears in exactly one split and in no earlier 12-node data. Presentations are random and noncanonical. Hop counts are balanced and interleaved before shuffling.',
        'tasks':{'pointer':{}}}
    for index,(split,(per_hop,hops)) in enumerate(design.items()):
        rng=random.Random(seed+index);rows=[]
        for i in range(per_hop*len(hops)):
            while True:
                row=pointer_record(rng,NODES,hops[i%len(hops)])
                if row['id'] not in seen and row['presentation']!=canonical:break
            seen.add(row['id']);row['split']=split;rows.append(row)
        rng.shuffle(rows)
        raw=''.join(canonical_json(r)+'\n' for r in rows).encode();path=directory/(split+'.jsonl');path.write_bytes(raw)
        manifest['tasks']['pointer'][split]={'path':str(path.relative_to(root)),'examples':len(rows),
            'sha256':hashlib.sha256(raw).hexdigest(),'steps':dict(sorted(Counter(r['steps'] for r in rows).items()))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_pointer_suite(root)
    return manifest


def validate_pointer_suite(root,check_prior=True):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());design=m['design']
    data={split:load_algorithmic_split(root,'pointer',split,128) for split in m['tasks']['pointer']}
    if set(data)!=set(design):raise ValueError('Splits differ from the declared design')
    for split,d in data.items():
        per_hop,hops=design[split]
        if any(len(r['successors'])!=NODES or r['presentation']==sorted(r['presentation']) for r in d.records):
            raise ValueError(f'{split}: wrong node count or canonical presentation')
        if Counter(r['steps'] for r in d.records)!=Counter({h:per_hop for h in hops}):raise ValueError(f'{split}: hop counts differ from the design')
    names=list(data)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if data[a].ids&data[b].ids:raise ValueError(f'Map overlap between {a} and {b}')
    if check_prior:
        prior,files=prior_inventory(m['prior_data_root'],root,NODES)
        if files!=m['prior_files'] or len(prior)!=m['excluded_prior_maps']:raise ValueError('Prior inventory changed')
        if any(d.ids&prior for d in data.values()):raise ValueError('Overlap with prior data')
    return data

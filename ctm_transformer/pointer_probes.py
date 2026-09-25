"""Exploratory pointer-format probes (development only; never paper endpoints).

Each probe renders a single 12-cycle map in random edge order, then asks
for the k-hop successor of QUERIES distinct nodes. Formats differ only in
how queries/answers and targets are written:

* ``block``:       ``map K>J;...;hops k;ask KCA...=JBE...``  (the dense calibration format)
* ``interleaved``: ``map K>J;...;hops k;ask K:J;C:B;...``   each answer directly follows its query
* ``lowercase``:   ``map K>j;...;hops k;ask K:j;C:b;...``   interleaved, with targets written in lowercase,
                   so sources and targets use disjoint symbols (standard associative recall at k=1)
* ``direct``:      ``map K>j;...;hops k;ask Kj;Cb;...``     lowercase targets; each answer is predicted at its query token
* ``mqar``:        ``map Kj;...;hops k;ask Kj;Cb;...``      as ``direct``, and each value directly follows its key (MQAR layout)
* ``mqar_random``: ``mqar`` layout, but each key's value is an independent random letter (repeats allowed),
                   so there is no permutation structure to exclude; only valid for one hop
* ``repeat``:      a random string of 8-24 distinct letters, ``;``, then the same string again; the second copy is
                   supervised (the canonical induction-head test; ignores the map, hop count and queries)
* ``copy``:        ``mqar`` layout, but each answer is its own query in lowercase: a pipeline sanity check needing no lookup

Only answer letters and a final EOS are supervised. The dataset has the
`DensePointerDataset` interface, so `dense_pointer.evaluate_dense` and runner
v3 work unchanged.
"""
import hashlib,json,random
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,content_hash
from ctm_transformer.dense_pointer import NODES,QUERIES,hop,labels

FORMATS=('block','interleaved','lowercase','direct','mqar','mqar_random','copy','repeat')


def repeat_record(rng,length=None):
    import string
    # A random length removes any fixed-offset copy solution; only content matching works.
    length=length or rng.randint(8,24)
    letters=rng.sample(string.ascii_uppercase,length);text=''.join(letters)+';'+''.join(letters)
    # Position 21 onward: each token after the first repeated one is predictable only by induction.
    supervised=list(range(length+2,2*length+1))
    return {'task':'pointer_probe_repeat','format':'repeat','steps':1,'queries':letters[1:],'answers':[text[i] for i in supervised],
            'text':text,'supervised':supervised,'difficulty':{'length':length,'steps':1},'id':content_hash(['repeat',text])}


def probe_record(rng,steps,fmt,nodes=NODES,queries=QUERIES):
    if fmt=='repeat':return repeat_record(rng)
    # Smaller maps (curriculum probes) use a random subset of the 12 letters.
    names=labels(nodes) if nodes==NODES else sorted(rng.sample(labels(NODES),nodes))
    queries=min(queries,nodes);cycle=rng.sample(names,nodes)
    successors={cycle[i]:cycle[(i+1)%nodes] for i in range(nodes)}
    if fmt=='mqar_random':
        if steps!=1:raise ValueError('mqar_random defines one-hop lookup only')
        successors={n:rng.choice(names) for n in names}
    presentation=rng.sample(names,nodes)
    while presentation==names:presentation=rng.sample(names,nodes)
    asked=rng.sample(names,queries);answers=[q if fmt=='copy' else hop(successors,q,steps) for q in asked]
    target=(lambda x:x.lower()) if fmt in ('lowercase','direct','mqar','mqar_random','copy') else (lambda x:x)
    arrow='' if fmt in ('mqar','mqar_random','copy') else '>'
    head='map '+';'.join(f'{n}{arrow}{target(successors[n])}' for n in presentation)+f';hops {steps};ask '
    if fmt=='block':
        text=head+''.join(asked)+'='+''.join(answers);first=len(head)+queries+1
        supervised=list(range(first,first+queries))
    elif fmt in ('interleaved','lowercase','direct','mqar','mqar_random','copy'):
        text=head;supervised=[];separator=':' if fmt in ('interleaved','lowercase') else ''
        for q,a in zip(asked,answers):
            text+=f'{q}{separator}';supervised.append(len(text));text+=target(a)+';'
    else:raise ValueError(f'Unknown probe format {fmt}')
    return {'task':f'pointer_probe_{fmt}','format':fmt,'successors':successors,'presentation':presentation,'steps':steps,
            'queries':asked,'answers':[target(a) for a in answers],'text':text,'supervised':supervised,
            'difficulty':{'nodes':nodes,'steps':steps},'id':content_hash(['map',successors])}


def write_probe_split(path,fmt,hops,count,seed,exclude=frozenset(),nodes=(NODES,NODES),queries=QUERIES):
    rng=random.Random(seed);rows=[];seen=set(exclude)
    for i in range(count):
        while True:
            row=probe_record(rng,hops[i%len(hops)],fmt,rng.randint(*nodes),queries)
            if row['id'] not in seen:break
        seen.add(row['id']);rows.append(row)
    rng.shuffle(rows)
    Path(path).write_text(''.join(canonical_json(r)+'\n' for r in rows))
    return {r['id'] for r in rows}


class ProbeDataset:
    """Supervises the declared answer characters plus a final EOS."""
    def __init__(self,path,tokenizer,seq_len):
        self.tokenizer=tokenizer;raw=Path(path).read_bytes()
        self.records=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        self.ids={r['id'] for r in self.records}
        if len(self.ids)!=len(self.records):raise ValueError('Duplicate map')
        xs,ys,lengths=[],[],[]
        for r in self.records:
            assert [r['text'][i] for i in r['supervised']]==r['answers']
            tokens=[tokenizer.bos_token]+tokenizer.encode(r['text'])+[tokenizer.eot_token]
            if len(tokens)-1>seq_len:raise ValueError('Example exceeds seq_len')
            x=tokens[:-1];y=[-100]*len(x)
            for i in r['supervised']:y[i]=tokens[i+1]
            y[-1]=tokenizer.eot_token
            lengths.append(len(x));xs.append(x+[tokenizer.pad_token]*(seq_len-len(x)));ys.append(y+[-100]*(seq_len-len(y)))
        self.inputs,self.targets,self.lengths=torch.tensor(xs),torch.tensor(ys),torch.tensor(lengths)
        self.metadata={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'examples':len(self.records),
            'max_input_length':max(lengths),'supervised_tokens':int((self.targets!=-100).sum()),'task':self.records[0]['task'],
            'role':'exploratory probe; development only'}

    def __len__(self):return len(self.records)

    def batch(self,index):
        lengths=self.lengths[index];width=int(lengths.max())
        return self.inputs[index,:width],self.targets[index,:width],int(lengths.sum())


class InOrderDataset(ProbeDataset):
    """Serves training batches in file order, ignoring the runner's shuffled indices.

    Used only for curriculum probes: the file is written phase by phase, so the
    k-th update sees the k-th block of batch_size records. Validation and
    evaluation datasets are ordinary ProbeDatasets.
    """
    def __init__(self,path,tokenizer,seq_len):
        super().__init__(path,tokenizer,seq_len);self.served=0

    def batch(self,index):
        count=len(index);start=self.served;self.served+=count
        if self.served>len(self):raise ValueError('Curriculum data exhausted; the file must hold exactly steps x batch records')
        return super().batch(torch.arange(start,start+count))


def write_curriculum_split(path,fmt,phases,batch,seed,exclude=frozenset(),queries=QUERIES):
    """phases: [(hops, updates)]; each phase is shuffled internally and written in phase order."""
    rng=random.Random(seed);seen=set(exclude);lines=[]
    for hops,updates in phases:
        rows=[]
        for i in range(updates*batch):
            while True:
                row=probe_record(rng,hops[i%len(hops)],fmt,NODES,queries)
                if row['id'] not in seen:break
            seen.add(row['id']);rows.append(row)
        rng.shuffle(rows);lines+=[canonical_json(r)+'\n' for r in rows]
    Path(path).write_text(''.join(lines))
    return seen-set(exclude)

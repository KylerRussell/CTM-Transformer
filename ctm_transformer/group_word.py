"""Group word problem (running products) as dense sequence labeling.

For a sequence of group elements g_1 ... g_n, the label at g_i's position is the
running product p_i = g_1·g_2·…·g_i. Elements are permutations. The product
composes left to right: ``a·b`` applies ``a`` first and then ``b``, so
``(a·b)[x] = b[a[x]]`` and ``p_i = p_{i-1}·g_i``.

    input : <BOS> g1 g2 … gn =
    target:   —   p1 p2 … pn <EOS>

No product ever appears in the input, so teacher forcing leaks no label. Groups:
Z2 (parity), S3 (smallest non-abelian) and A5 (smallest non-solvable; its word
problem is NC1-complete). Elements map to single tokenizer characters in a fixed
canonical order (sorted permutation tuples).

Short sequences necessarily repeat (S3 has only 6 sequences of length 1), so
exact-sequence disjointness between splits is enforced, and exposed as ``ids``,
only for sequences of length >= MIN_HELD_OUT.
"""
import hashlib,itertools,json,random
from pathlib import Path
import torch
from ctm_transformer.algorithmic import AlgorithmicTokenizer,canonical_json,content_hash

SYMBOLS='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
MIN_HELD_OUT=8
TASK='group_word'


def _even(p):return sum(1 for i in range(len(p)) for j in range(i+1,len(p)) if p[i]>p[j])%2==0


def group_elements(name):
    if name=='Z2':return sorted(itertools.permutations(range(2)))
    if name=='S3':return sorted(itertools.permutations(range(3)))
    if name=='A5':return sorted(p for p in itertools.permutations(range(5)) if _even(p))
    raise ValueError(f'Unknown group {name}')


def compose(a,b):
    """a·b: apply a, then b."""
    return tuple(b[a[x]] for x in range(len(a)))


class Group:
    def __init__(self,name):
        self.name=name;self.elements=group_elements(name)
        if len(self.elements)>len(SYMBOLS):raise ValueError('Group too large for the symbol set')
        self.symbol={e:SYMBOLS[i] for i,e in enumerate(self.elements)};self.element={v:k for k,v in self.symbol.items()}

    def running_products(self,word):
        out=[];p=None
        for ch in word:
            g=self.element[ch];p=g if p is None else compose(p,g);out.append(self.symbol[p])
        return ''.join(out)


def group_record(rng,group,length):
    word=''.join(rng.choice(SYMBOLS[:len(group.elements)]) for _ in range(length))
    return {'task':TASK,'group':group.name,'word':word,'products':group.running_products(word),
            'steps':length,'difficulty':{'group':group.name,'length':length},'id':content_hash([TASK,group.name,word])}


def validate_group_record(record,group):
    if record['task']!=TASK or record['group']!=group.name or record['steps']!=len(record['word']):raise ValueError('Not a group-word record')
    if record['products']!=group.running_products(record['word']):raise ValueError('Wrong running products')
    if record['id']!=content_hash([TASK,group.name,record['word']]):raise ValueError('Inconsistent identity')


def write_group_split(path,group,lengths,count,seed,exclude=frozenset()):
    """Lengths are balanced by cycling; sequences of length >= MIN_HELD_OUT avoid `exclude`."""
    rng=random.Random(seed);rows=[]
    for i in range(count):
        while True:
            row=group_record(rng,group,lengths[i%len(lengths)])
            if row['steps']<MIN_HELD_OUT or row['id'] not in exclude:break
        rows.append(row)
    rng.shuffle(rows)
    Path(path).write_text(''.join(canonical_json(r)+'\n' for r in rows))
    return {r['id'] for r in rows if r['steps']>=MIN_HELD_OUT}


class GroupWordDataset:
    """Sequence-labeling batches with the runner's dataset interface."""
    def __init__(self,path,tokenizer,seq_len):
        self.tokenizer=tokenizer;raw=Path(path).read_bytes()
        self.records=[json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        groups={r['group'] for r in self.records}
        if len(groups)!=1:raise ValueError('Use one group per dataset')
        group=Group(groups.pop());xs,ys,lengths=[],[],[]
        for r in self.records:
            validate_group_record(r,group)
            x=[tokenizer.bos_token]+tokenizer.encode(r['word']+'=')
            if len(x)>seq_len:raise ValueError('Example exceeds seq_len')
            y=[-100]+tokenizer.encode(r['products'])+[tokenizer.eot_token]
            lengths.append(len(x));xs.append(x+[tokenizer.pad_token]*(seq_len-len(x)));ys.append(y+[-100]*(seq_len-len(y)))
        self.inputs,self.targets,self.lengths=torch.tensor(xs),torch.tensor(ys),torch.tensor(lengths)
        # Held-out identities: only sequences long enough to be enforced disjoint across splits.
        self.ids={r['id'] for r in self.records if r['steps']>=MIN_HELD_OUT}
        self.metadata={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'examples':len(self.records),
            'max_input_length':max(lengths),'supervised_tokens':int((self.targets!=-100).sum()),'task':f'{TASK}_{group.name}',
            'lengths':sorted({r['steps'] for r in self.records})}

    def __len__(self):return len(self.records)

    def batch(self,index):
        lengths=self.lengths[index];width=int(lengths.max())
        return self.inputs[index,:width],self.targets[index,:width],int(lengths.sum())

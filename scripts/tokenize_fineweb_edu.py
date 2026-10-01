"""Train a 32k byte-level BPE tokenizer on FineWeb-Edu and tokenize the 10BT sample into uint16 shards.

* The tokenizer is trained on documents of the first source file only.
* The validation split is the last VALIDATION_DOCS documents of the last source file. Those documents are never trained on.
* Every document is preceded by <|endoftext|> (id 0), as in GPT-2-style corpora.
* Each source file becomes one train shard `train_XXX.bin` (uint16). The manifest
  records token counts, the tokenizer hash and the source snapshot.
"""
import hashlib,json,os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer,decoders,models,pre_tokenizers,trainers

OUT=Path('research/data/pretrain/fineweb_edu_32k');SOURCE=Path('research/data/pretrain/fineweb_edu_10bt_source.json')
VOCAB=32768;TOKENIZER_DOCS=1_000_000;VALIDATION_DOCS=50_000;EOT='<|endoftext|>'


def texts(path,limit=None,batch=10_000):
    seen=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch,columns=['text']):
        for t in b.column(0).to_pylist():
            yield t;seen+=1
            if limit and seen>=limit:return


def train_tokenizer(first_file):
    tok=Tokenizer(models.BPE());tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False);tok.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=VOCAB,special_tokens=[EOT],initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),min_frequency=2,show_progress=False)
    tok.train_from_iterator(texts(first_file,TOKENIZER_DOCS),trainer=trainer,length=TOKENIZER_DOCS)
    tok.save(str(OUT/'tokenizer.json'));return tok


def encode_file(job):
    path,out,skip_last=job;os.environ['TOKENIZERS_PARALLELISM']='false'
    tok=Tokenizer.from_file(str(OUT/'tokenizer.json'));eot=tok.token_to_id(EOT)
    total=pq.ParquetFile(path).metadata.num_rows;keep=total-skip_last;parts=[];seen=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=2_000,columns=['text']):
        batch=b.column(0).to_pylist()[:max(0,keep-seen)];seen+=len(b)
        for enc in tok.encode_batch(batch):parts.append(np.array([eot]+enc.ids,dtype=np.uint16))
        if seen>=keep:break
    arr=np.concatenate(parts);arr.tofile(OUT/out);return out,int(len(arr)),min(keep,total)


def encode_validation(path):
    tok=Tokenizer.from_file(str(OUT/'tokenizer.json'));eot=tok.token_to_id(EOT)
    total=pq.ParquetFile(path).metadata.num_rows;docs=[];seen=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=10_000,columns=['text']):
        rows=b.column(0).to_pylist()
        docs+=rows[max(0,total-VALIDATION_DOCS-seen):];seen+=len(rows)
    arr=np.concatenate([np.array([eot]+e.ids,dtype=np.uint16) for e in tok.encode_batch(docs)]);arr.tofile(OUT/'validation_000.bin')
    return int(len(arr)),len(docs)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True);source=json.loads(SOURCE.read_text());files=source['files']
    tok=train_tokenizer(files[0]) if not (OUT/'tokenizer.json').exists() else Tokenizer.from_file(str(OUT/'tokenizer.json'))
    assert tok.get_vocab_size()==VOCAB and tok.token_to_id(EOT)==0
    jobs=[(f,f'train_{i:03d}.bin',VALIDATION_DOCS if i==len(files)-1 else 0) for i,f in enumerate(files)]
    with Pool(len(jobs)) as pool:train=pool.map(encode_file,jobs)
    val_tokens,val_docs=encode_validation(files[-1])
    manifest={'name':'fineweb_edu_32k','source':source,'tokenizer':'tokenizer.json','tokenizer_sha256':sha(OUT/'tokenizer.json'),'vocab_size':VOCAB,
        'eot_id':0,'dtype':'uint16','tokenizer_training_docs':f'first {TOKENIZER_DOCS} documents of {Path(files[0]).name}',
        'train':[t[0] for t in train],'train_tokens':{t[0]:t[1] for t in train},'train_documents':{t[0]:t[2] for t in train},
        'validation':['validation_000.bin'],'validation_tokens':val_tokens,'validation_documents':val_docs,
        'validation_rule':f'last {VALIDATION_DOCS} documents of {Path(files[-1]).name}, excluded from train_{len(files)-1:03d}.bin',
        'total_train_tokens':sum(t[1] for t in train)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ('total_train_tokens','validation_tokens','validation_documents')}))

if __name__=='__main__':main()

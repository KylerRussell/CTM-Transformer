"""Code and math data for the stage-0 probe (research/LADDER_PLAN.md, 2026-10-28), in the FineWeb-Edu 32k tokenization.

Sources (public, downloaded anonymously from the Hugging Face hub; revisions recorded in the manifest):
- math: HuggingFaceTB/finemath, finemath-4plus. Training text from train-00000-of-00064, held-out text from
  train-00063-of-00064 (a different file, so no document is in both).
- code: codeparrot/codeparrot-clean (Python files). Training text from file-000000000001, held-out text from
  file-000000000054.
- GSM8K test (openai/gsm8k, main), MBPP test (google-research-datasets/mbpp, full) and HumanEval
  (openai/openai_humaneval): prompt / solution pairs, written as JSON for scripts/code_math_probe.py.

Outputs in research/data/pretrain/code_math_32k/ (tokenizer: research/data/pretrain/fineweb_edu_32k/tokenizer.json):
- math_train.bin and code_train.bin: exactly TRAIN_TOKENS tokens each, so that the trainer's uniform window
  permutation over both shards gives a 50/50 mix; every document is preceded by <|endoftext|> (id 0).
- math_valid.bin and code_valid.bin: VALID_TOKENS tokens each, from the held-out files.
- tasks.json: GSM8K / MBPP / HumanEval prompts and reference solutions.
- manifest.json: the trainer's data manifest (train = both train shards, validation = both held-out shards).

Usage: python -m scripts.code_math_data
"""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

OUT=Path('research/data/pretrain/code_math_32k');TOKENIZER=Path('research/data/pretrain/fineweb_edu_32k/tokenizer.json')
TRAIN_TOKENS=60_000_000;VALID_TOKENS=2_000_000
MATH=('HuggingFaceTB/finemath','finemath-4plus/train-00000-of-00064.parquet','finemath-4plus/train-00063-of-00064.parquet')
CODE=('codeparrot/codeparrot-clean','file-000000000001.json.gz','file-000000000054.json.gz')


def fetch(repo,name):return hf_hub_download(repo,name,repo_type='dataset',token=False)


def parquet_texts(path):
    for b in pq.ParquetFile(path).iter_batches(batch_size=2_000,columns=['text']):yield from b.column(0).to_pylist()


def jsonl_gz_texts(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)['content']


def tokenize(texts,limit,out,tok):
    parts,total,docs,batch=[],0,0,[]
    def flush():
        nonlocal total,docs
        for e in tok.encode_batch(batch):
            parts.append(np.array([0]+e.ids,dtype=np.uint16));total+=len(e.ids)+1;docs+=1
        batch.clear()
    for t in texts:
        batch.append(t)
        if len(batch)==2_000:
            flush()
            if total>=limit:break
    if total<limit and batch:flush()
    arr=np.concatenate(parts)
    if len(arr)<limit:raise ValueError(f'{out}: only {len(arr)} tokens')
    arr[:limit].tofile(OUT/out);return {'tokens':limit,'documents':docs}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks():
    g=pq.read_table(fetch('openai/gsm8k','main/test-00000-of-00001.parquet')).to_pylist()
    m=pq.read_table(fetch('google-research-datasets/mbpp','full/test-00000-of-00001.parquet')).to_pylist()
    h=pq.read_table(fetch('openai/openai_humaneval','openai_humaneval/test-00000-of-00001.parquet')).to_pylist()
    return {'gsm8k':[{'prompt':f'Question: {r["question"]}\nAnswer:','solution':' '+r['answer']} for r in g],
            'mbpp':[{'prompt':f'"""\n{r["text"]}\n{r["test_list"][0]}\n"""\n','solution':r['code'].replace('\r\n','\n')} for r in m],
            'humaneval':[{'prompt':r['prompt'],'solution':r['canonical_solution']} for r in h]}


def main():
    OUT.mkdir(parents=True,exist_ok=True);tok=Tokenizer.from_file(str(TOKENIZER));assert tok.token_to_id('<|endoftext|>')==0
    stats={}
    for name,(repo,train,valid),reader in (('math',MATH,parquet_texts),('code',CODE,jsonl_gz_texts)):
        stats[f'{name}_train.bin']=tokenize(reader(fetch(repo,train)),TRAIN_TOKENS,f'{name}_train.bin',tok)
        stats[f'{name}_valid.bin']=tokenize(reader(fetch(repo,valid)),VALID_TOKENS,f'{name}_valid.bin',tok)
        print(name,stats[f'{name}_train.bin'],stats[f'{name}_valid.bin'],flush=True)
    (OUT/'tasks.json').write_text(json.dumps(tasks())+'\n')
    from huggingface_hub import HfApi
    api=HfApi(token=False);revisions={r:api.dataset_info(r,token=False).sha for r in (MATH[0],CODE[0],'openai/gsm8k','google-research-datasets/mbpp','openai/openai_humaneval')}
    manifest={'name':'code_math_32k','sources':{'math':MATH,'code':CODE},'revisions':revisions,'tokenizer':'../fineweb_edu_32k/tokenizer.json',
              'tokenizer_sha256':sha(TOKENIZER),'vocab_size':tok.get_vocab_size(),'eot_id':0,'dtype':'uint16',
              'train':['math_train.bin','code_train.bin'],'validation':['math_valid.bin','code_valid.bin'],'shards':stats,'tasks':'tasks.json',
              'tasks_sha256':sha(OUT/'tasks.json')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(stats))


if __name__=='__main__':main()

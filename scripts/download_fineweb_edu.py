"""Download the FineWeb-Edu 10BT sample (14 parquet files) into the Hugging Face cache and record the snapshot revision."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO='HuggingFaceFW/fineweb-edu';PATTERN='sample/10BT/*.parquet'
OUT=Path('research/data/pretrain/fineweb_edu_10bt_source.json')

if __name__=='__main__':
    path=snapshot_download(REPO,repo_type='dataset',allow_patterns=[PATTERN],max_workers=8)
    files=sorted(str(p) for p in Path(path).glob('sample/10BT/*.parquet'))
    OUT.write_text(json.dumps({'repo':REPO,'pattern':PATTERN,'snapshot':path,'revision':Path(path).name,'files':files},indent=2)+'\n')
    print(f'{len(files)} files in {path}')

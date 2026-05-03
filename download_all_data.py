import os
from huggingface_hub import snapshot_download

# The specific subsets from your curriculum
datasets = [
    ("nvidia/Nemotron-Pretraining-Specialized-v1", [
        "Nemotron-Pretraining-RQA",
        "Nemotron-Pretraining-InfiniByte-Reasoning",
        "Nemotron-Pretraining-Wiki-Rewrite",
        "Nemotron-Pretraining-Scientific-Coding",
        "Nemotron-Pretraining-Math-Textbooks",
        "Nemotron-Pretraining-STEM-SFT"
    ]),
    ("nvidia/Nemotron-Pretraining-Specialized-v1.1", [
        "Nemotron-Pretraining-Code-Concepts",
        "Nemotron-Pretraining-Unconditional-Algorithmic",
        "Nemotron-Pretraining-Formal-Logic",
        "Nemotron-Pretraining-Economics",
        "Nemotron-Pretraining-Multiple-Choice"
    ]),
    ("nvidia/Nemotron-CC-Math-v1", ["4plus_MIND"])
]

local_dir = os.path.abspath("./data_cache")
os.makedirs(local_dir, exist_ok=True)

for repo_id, subsets in datasets:
    for subset in subsets:
        print(f"\n--- Downloading {repo_id} | Subset: {subset} ---")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{subset}/*.parquet",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

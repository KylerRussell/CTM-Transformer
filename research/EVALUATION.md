# Evaluation validation

This milestone validates scoring and result provenance. It does not establish model quality, compare architectures, or reproduce the historical benchmark scores: no trained checkpoint was present in this checkout.

## Corrected behavior

The adapter now tokenizes context and continuation jointly, moving trailing context whitespace to the continuation before splitting the token sequence. Empty contexts receive a recorded prefix token, so the first answer token is scored. Empty continuations have log probability zero and an exact-match flag of true.

Only context may be truncated. The model receives up to `max_seq_len` input tokens and predicts the complete continuation, including the token immediately following the final input. Continuations longer than that window raise an error instead of producing a partial score. Boundaries that fall inside a merged BPE token also raise an explicit error: splitting a word there does not define the same token-level conditional likelihood as a word-aligned prompt.

Right-padded batches score only real continuation positions. Log-softmax and likelihood accumulation use FP32, including when model execution uses BF16. The adapter checks finite scores and vocabulary compatibility. It supports conditional log-likelihood tasks; generation and rolling corpus perplexity remain unimplemented and raise errors.

Checkpoint loading requires complete, matching state dictionaries. It strips repeated DDP/compile prefixes, rejects collisions, missing/unexpected parameters, unknown configuration fields, and inconsistent tied weights. It hashes and deserializes the same open file and rejects a checkpoint modified in place while loading. Older checkpoint schemas require an explicit migration; partial weights are never treated as a valid evaluation model.

## Verification

The 27 tests in `tests/test_eval_harness.py` cover:

- Closed-form bigram probabilities and greedy flags, independent of the adapter's tensor indexing.
- Empty contexts/requests/continuations, whitespace-sensitive BPE, and merged-token boundaries.
- Left truncation and answers exactly filling the window; rejection of oversized answers.
- Actual CTM scores versus one-prefix-at-a-time evaluation, in FP32 and BF16, with batching/padding checks.
- Actual CTM logits through the pinned harness's reference tokenization and HuggingFace likelihood implementation.
- Strict checkpoint round trips, wrapper-prefix cleanup, and invalid checkpoint rejection.
- Reconstructing tiktoken and HuggingFace fast tokenizers from the saved rules.
- Dataset fingerprints and source hashes.
- The complete CLI path on a local two-example multiple-choice task, including two thought budgets, task loading, aggregation, sample logging, and output JSON.

The reference algorithms come from [EleutherAI's TemplateLM](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/api/model.py) and [HuggingFace adapter](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/models/huggingface.py). Tests use the installed reference code directly on the same CTM logits; no pretrained reference-model weights are downloaded. GPU-dependent checks run on the RTX 3090s. Test reports are in `results/eval_checks_gpu0.xml` and `results/eval_checks_gpu1.xml`.

## Environment and commands

Activate the GPU environment as described in `README.md`, then install the evaluation dependency pins:

```sh
python -m pip install -r research/requirements-eval.txt
```

The tested harness is 0.4.9.1, Transformers 4.51.3, and PEFT 0.15.2. Pinning avoids an import incompatibility with Transformers 5 and a local-task-path logging bug encountered in lm-eval 0.4.9.2. `evaluation-environment.txt` records the complete environment; the earlier `environment.txt` preserves the original model-preflight environment. Install the CUDA PyTorch wheel as described in `README.md` before restoring either package snapshot.

Run the two test groups independently on the two GPUs:

```sh
CTM_TEST_DEVICE=cuda:0 python -m pytest tests/test_eval_harness.py \
  -k 'analytic or continuation or empty or bpe or prefix or matches' -q
```

```sh
CTM_TEST_DEVICE=cuda:1 python -m pytest tests/test_eval_harness.py \
  -k 'not (analytic or continuation or empty or bpe or prefix or matches)' -q
```

Once a trained checkpoint is available, run a small evaluation before a full task suite:

```sh
python -m scripts.eval_harness \
  --checkpoint path/to/checkpoint.pt \
  --tasks lambada_openai,hellaswag,piqa,arc_easy \
  --device cuda:0 --batch_size 4 --dtype bfloat16 \
  --t_sweep 1,2,4 --limit 100 --seed 17 \
  --output research/results/checkpoint_pilot.eval.json
```

Select thought budgets supported by that checkpoint. Remove `--limit` for the full benchmark. `--max_seq_len` defaults to the smaller of the saved training sequence length and model window; an explicit override is recorded. A tokenizer override is explicit and recorded; `--tokenizer_revision` can pin a HuggingFace tokenizer commit. Fast HuggingFace tokenizers or tiktoken are required so the exact tokenization rules can be saved. `--prefix_token_id` is available if no EOS/BOS token can be inferred. `--include_path` adds local task YAML files.

## Result format and reproducibility

Outputs now use **schema version 2**, rather than the old bare map of thought budgets to metrics:

- `results["4"]`: aggregate metrics at T=4 (equivalent in purpose to the old top-level `"4"` entry).
- `evaluations["4"]`: full harness output, including per-example inputs, responses, task definitions, versions, and few-shot settings.
- `metadata`: checkpoint SHA-256 and saved training counters, original/effective model configurations, tokenizer identity and snapshot hash, package versions, source hashes/commit/dirty status, device/precision, seed, batch size, sequence length, task names, example limit, and loaded dataset split fingerprints/versions/row counts.
- `complete`: false until every requested thought budget finishes. Results are written atomically after each completed budget.

The adjacent `.tokenizer.json` stores complete tiktoken merge rules/pattern/special tokens or the HuggingFace fast-tokenizer backend and special tokens. Archive this file with the evaluation JSON, checkpoint, package snapshot, and source revision. If `--output` is omitted, the result is written beside the checkpoint as `<stem>.eval.json`.

Dataset fingerprints and recorded versions identify the loaded splits; they are not a substitute for retaining a dataset manifest or immutable source revision. A missing upstream revision is recorded as missing, rather than inferred. The saved per-example documents, prompts, and responses preserve what was actually evaluated. The new scoring policy and earlier model fixes mean historical scores should be rerun before they are used as research evidence.

## Baseline integration

The same likelihood adapter and strict loader now support the standard Transformer and recurrent-depth adaptation described in [BASELINES.md](BASELINES.md). New training checkpoints carry `model_family`; legacy checkpoints without it are interpreted as CTM. Baseline configurations must be fully specified, and the saved family must agree with the configuration. Metadata records the family. Use T=1 for the standard Transformer; larger thought budgets are rejected rather than silently ignored. Baseline strict reloads and adapter probabilities are verified in `tests/test_baselines.py`; all 27 original evaluation tests passed after integration.

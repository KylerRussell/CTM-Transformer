"""Evaluate CTM and baseline checkpoints with lm-eval 0.4.9.1 and auditable output.

Run from the repository root, for example:
  python -m scripts.eval_harness --checkpoint model.pt --t_sweep 1,4,8

Scoring preserves all continuation tokens. Only context is left-truncated;
continuations longer than the model input window are rejected. Outputs use
schema version 2: metadata, per-depth results, and full harness evaluations
(including per-example responses and task definitions).
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
from typing import Optional

import torch
import torch.nn.functional as F
from lm_eval.api.model import LM


class CTMTransformerLM(LM):
    """Causal likelihood adapter; tokenization uses no automatic BOS/EOS."""

    def __init__(self, model, tokenizer, device="cuda:0", batch_size=4,
                 max_thought_steps: Optional[int] = None, max_seq_len=512,
                 prefix_token_id: Optional[int] = None):
        super().__init__()
        if batch_size <= 0 or max_seq_len <= 0:
            raise ValueError("batch_size and max_seq_len must be positive")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_thought_steps = max_thought_steps
        self._max_seq_len = max_seq_len
        if max_seq_len > model.config.max_seq_len:
            raise ValueError("max_seq_len exceeds the model's configured input window")
        inner = getattr(tokenizer, "tokenizer", tokenizer)
        if prefix_token_id is None:
            for owner, field in ((tokenizer, "eot_token"), (inner, "eos_token_id"),
                                 (inner, "bos_token_id")):
                prefix_token_id = getattr(owner, field, None)
                if prefix_token_id is not None:
                    break
        if prefix_token_id is None:
            raise ValueError("Tokenizer has no EOS/BOS token; specify prefix_token_id explicitly")
        self._eot_token_id = int(prefix_token_id)
        if not 0 <= self._eot_token_id < model.config.vocab_size:
            raise ValueError("prefix_token_id is outside the model vocabulary")
        vocab_size = getattr(tokenizer, "n_vocab", None)
        if vocab_size is not None and vocab_size > model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary is larger than the checkpoint vocabulary")

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def prefix_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_seq_len

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string, **kwargs):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context, continuation):
        # Moving whitespace retains its BPE association with the next word.
        trimmed = context.rstrip()
        continuation = context[len(trimmed):] + continuation
        context_tokens = self.tok_encode(trimmed) if trimmed else []
        joined = self.tok_encode(trimmed + continuation)
        if joined[:len(context_tokens)] != context_tokens:
            raise ValueError(
                "Context/continuation boundary falls inside a merged token. "
                "Use a token-aligned boundary (normally put the word-leading space "
                "in the continuation); an exact conditional score is ambiguous here."
            )
        continuation_tokens = joined[len(context_tokens):]
        if not context_tokens:
            context_tokens = [self.prefix_token_id]
            # Match the reference harness's explicitly supplied prefix convention.
            if continuation_tokens and continuation_tokens[0] == self.prefix_token_id:
                continuation_tokens = continuation_tokens[1:]
        return context_tokens, continuation_tokens

    @torch.no_grad()
    def _model_call(self, ids):
        return self.model(ids, max_thought_steps=self._max_thought_steps)["logits"]

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm=False):
        pairs = [request.args if hasattr(request, "args") else request for request in requests]
        results = [None] * len(pairs)
        encoded = []
        for index, (context, continuation) in enumerate(pairs):
            # The probability of an empty continuation is 1.
            if continuation == "":
                results[index] = (0.0, True)
                continue
            ctx, cont = self._encode_pair(context, continuation)
            if not cont:
                results[index] = (0.0, True)
                continue
            if len(cont) > self.max_length:
                raise ValueError(
                    f"Continuation has {len(cont)} tokens, exceeding max_seq_len="
                    f"{self.max_length}; refusing to discard scored tokens."
                )
            if min(ctx + cont) < 0 or max(ctx + cont) >= self.model.config.vocab_size:
                raise ValueError("Token ID is outside the checkpoint vocabulary")
            # The last target is never an input. Keep one context token even when
            # the continuation occupies the entire input window.
            inputs = (ctx + cont)[-(self.max_length + 1):][:-1]
            encoded.append((index, inputs, cont))
        encoded.sort(key=lambda item: -len(item[1]))
        for offset in range(0, len(encoded), self.batch_size):
            batch = encoded[offset:offset + self.batch_size]
            width = max(len(inputs) for _, inputs, _ in batch)
            ids = torch.full((len(batch), width), self.eot_token_id,
                             dtype=torch.long, device=self.device)
            for row, (_, inputs, _) in enumerate(batch):
                ids[row, :len(inputs)] = torch.tensor(inputs, device=self.device)
            # Right padding is causally inaccessible to all scored positions.
            log_probs = F.log_softmax(self._model_call(ids).float(), dim=-1)
            for row, (index, inputs, cont) in enumerate(batch):
                predictions = log_probs[row, len(inputs) - len(cont):len(inputs)]
                targets = torch.tensor(cont, dtype=torch.long, device=self.device)
                score = predictions.gather(-1, targets[:, None]).sum()
                if not torch.isfinite(score):
                    raise RuntimeError("Non-finite continuation log-likelihood")
                results[index] = (float(score), bool((predictions.argmax(-1) == targets).all()))
        for pair, result in zip(pairs, results):
            self.cache_hook.add_partial("loglikelihood", pair, result)
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm=False):
        raise NotImplementedError("Rolling corpus likelihood is not implemented; use conditional likelihood tasks")

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError("Generation tasks are not implemented in this adapter")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(checkpoint_path, config_overrides=None):
    """Load local training checkpoints strictly; never evaluate partial weights."""
    from ctm_transformer.research import config_from_dict, build_model

    checkpoint_path = Path(checkpoint_path)
    # Training checkpoints include configuration objects and optimizer metadata.
    # Hash and deserialize the same open file, even if a training process
    # replaces a `latest.pt` path during evaluation.
    with checkpoint_path.open("rb") as source:
        before = os.fstat(source.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        source.seek(0)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading; evaluate a stable snapshot")
    if not isinstance(checkpoint, dict) or "config" not in checkpoint:
        raise ValueError("Checkpoint must include its training config")
    saved_config = checkpoint["config"]
    config_dict = dict(saved_config if isinstance(saved_config, dict) else vars(saved_config))
    if config_overrides:
        config_dict.update(config_overrides)
    family = checkpoint.get("model_family", "ctm")
    try:
        config = config_from_dict(config_dict, family, require_all=family != 'ctm')
    except ValueError as error:
        raise ValueError(f"Unknown or invalid checkpoint configuration fields require explicit migration: {error}") from error
    state = next((checkpoint[key] for key in ("model", "model_state_dict", "state_dict")
                  if key in checkpoint), None)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain a named model state dictionary")
    fixed = {}
    for name, tensor in state.items():
        clean = name
        while clean.startswith(("module.", "_orig_mod.")):
            clean = clean.split(".", 1)[1]
        if clean in fixed:
            raise ValueError(f"Checkpoint wrapper-prefix collision for {clean}")
        fixed[clean] = tensor
    if config.tie_embeddings and (family != "ctm" or config.use_shared_head_film):
        embedding, head = fixed.get("token_embedding.weight"), fixed.get("lm_head.weight")
        if embedding is not None and head is not None and not torch.equal(embedding, head):
            raise ValueError("Checkpoint contains inconsistent tied embedding/head weights")
    model = build_model(config)
    model.load_state_dict(fixed, strict=True)
    model._checkpoint_metadata = {
        "model_family": family,
        "path": str(checkpoint_path.resolve()), "sha256": digest.hexdigest(),
        "size_bytes": after.st_size,
        "training_config": saved_config if isinstance(saved_config, dict) else vars(saved_config),
        "config_overrides": config_overrides or {},
        "training_counters": {key: checkpoint[key] for key in
                              ("step", "tokens_seen", "tokens_processed", "total_tokens") if key in checkpoint},
    }
    return model, config


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_json_default) + "\n")
    temporary.replace(path)


def _tokenizer_snapshot(tokenizer):
    """Save effective tokenization rules, not only a mutable model name."""
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(inner, "_mergeable_ranks"):
        return {
            "kind": "tiktoken", "name": inner.name, "pattern": inner._pat_str,
            "mergeable_ranks": [[base64.b64encode(token).decode("ascii"), rank]
                                for token, rank in sorted(inner._mergeable_ranks.items(), key=lambda x: x[1])],
            "special_tokens": inner._special_tokens,
        }
    if hasattr(inner, "backend_tokenizer"):
        return {"kind": "huggingface_fast", "backend": json.loads(inner.backend_tokenizer.to_str()),
                "special_tokens_map": inner.special_tokens_map,
                "name_or_path": inner.name_or_path,
                "revision": inner.init_kwargs.get("_commit_hash"), "add_special_tokens": False}
    # A vocabulary alone cannot reproduce a slow tokenizer's segmentation rules.
    raise ValueError("Evaluation provenance requires tiktoken or a HuggingFace fast tokenizer")


def _dataset_metadata(task_dict):
    records = {}
    for name, task in task_dict.items():
        if isinstance(task, dict):
            records.update(_dataset_metadata(task))
            continue
        splits = {}
        for split, dataset in getattr(task, "dataset", {}).items():
            info = getattr(dataset, "info", None)
            splits[str(split)] = {
                "fingerprint": getattr(dataset, "_fingerprint", None),
                "num_rows": getattr(dataset, "num_rows", None),
                "version": str(info.version) if info and info.version is not None else None,
                "dataset_name": getattr(info, "dataset_name", None),
            }
        config = task.dump_config()
        records[str(name)] = {"dataset_path": config.get("dataset_path"),
                              "dataset_name": config.get("dataset_name"),
                              "dataset_kwargs": config.get("dataset_kwargs"), "splits": splits}
    return records


def _code_metadata():
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, None
    paths = [Path(__file__), *sorted((root / "ctm_transformer").glob("*.py"))]
    return {"commit": commit, "worktree_status": status,
            "source_sha256": {str(p.relative_to(root)): _sha256_file(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", help="Explicit tokenizer override; recorded in provenance")
    parser.add_argument("--tokenizer_revision", help="HuggingFace tokenizer revision (prefer a commit SHA)")
    parser.add_argument("--tasks", default="lambada_openai,hellaswag,piqa,arc_easy")
    parser.add_argument("--include_path", help="Directory containing additional lm-eval task definitions")
    parser.add_argument("--thought_steps", type=int)
    parser.add_argument("--t_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, help="Default: minimum of training and model window sizes")
    parser.add_argument("--prefix_token_id", type=int)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, help="Default: <checkpoint>.eval.json")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    model, config = _load_checkpoint(args.checkpoint)
    model.to(torch.device(args.device), getattr(torch, args.dtype)).eval()
    tokenizer_name = args.tokenizer or config.tokenizer
    if tokenizer_name.startswith("hf:"):
        from transformers import AutoTokenizer
        from ctm_transformer.train import HFTokenizerWrapper
        tokenizer = HFTokenizerWrapper(AutoTokenizer.from_pretrained(
            tokenizer_name[3:], revision=args.tokenizer_revision, use_fast=True,
            trust_remote_code=False,
        ))
    else:
        if args.tokenizer_revision:
            parser.error("tokenizer_revision only applies to HuggingFace tokenizers")
        import tiktoken
        tokenizer = tiktoken.get_encoding(tokenizer_name)
    output = args.output or Path(args.checkpoint).with_suffix(".eval.json")
    tokenizer_path = output.with_suffix(".tokenizer.json")
    if Path(args.checkpoint).resolve() in (output.resolve(), tokenizer_path.resolve()):
        parser.error("evaluation output must not overwrite the checkpoint")
    _write_json(tokenizer_path, _tokenizer_snapshot(tokenizer))
    t_values = ([int(t.strip()) for t in args.t_sweep.split(",") if t.strip()]
                if args.t_sweep is not None else [args.thought_steps or config.max_thought_steps])
    if not t_values or any(t <= 0 for t in t_values) or len(set(t_values)) != len(t_values):
        parser.error("thought budgets must be nonempty, positive, and unique")
    if args.thought_steps is not None and args.thought_steps <= 0:
        parser.error("thought_steps must be positive")
    if max(t_values) > config.max_thought_steps and (
        any(getattr(config, key, False) for key in ("per_tick_heads", "use_shared_head_film", "use_loop_pos_emb"))
    ):
        parser.error("thought budget exceeds max_thought_steps for tick-indexed parameters")
    if getattr(config, 'model_family', 'ctm') == 'transformer' and t_values != [1]:
        parser.error('Standard Transformer has fixed layer depth; use T=1')
    max_seq_len = args.max_seq_len if args.max_seq_len is not None else min(config.max_seq_len, config.seq_len)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        parser.error("at least one task is required")
    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager

    class RecordingTaskManager(TaskManager):
        def load_task_or_group(self, task_list):
            self.loaded_tasks = super().load_task_or_group(task_list)
            return self.loaded_tasks

    manager = RecordingTaskManager(include_path=args.include_path)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": model._checkpoint_metadata, "effective_config": asdict(config),
        "tokenizer": {"identifier": tokenizer_name, "override": args.tokenizer,
                      "requested_revision": args.tokenizer_revision, "snapshot": str(tokenizer_path.resolve()),
                      "sha256": _sha256_file(tokenizer_path)},
        "code": _code_metadata(),
        "packages": {name: version(name) for name in ("torch", "lm-eval", "transformers", "datasets", "tiktoken")},
        "device": args.device, "dtype": args.dtype, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if torch.device(args.device).type == "cuda" else None,
        "batch_size": args.batch_size, "max_seq_len": max_seq_len, "seed": args.seed,
        "tasks": tasks, "include_path": args.include_path, "num_fewshot": args.num_fewshot, "limit": args.limit, "thought_budgets": t_values,
        "scoring": "joint tokenization; explicit prefix; context-only truncation; FP32 log_softmax",
    }
    report = {"schema_version": 2, "metadata": metadata, "results": {}, "evaluations": {}, "complete": False}
    for T in t_values:
        lm = CTMTransformerLM(model, tokenizer, args.device, args.batch_size, T, max_seq_len, args.prefix_token_id)
        metadata["prefix_token_id"] = lm.prefix_token_id
        evaluated = simple_evaluate(
            model=lm, tasks=tasks, task_manager=manager, num_fewshot=args.num_fewshot,
            limit=args.limit, batch_size=args.batch_size, log_samples=True,
            random_seed=args.seed, numpy_random_seed=args.seed, torch_random_seed=args.seed,
            fewshot_random_seed=args.seed,
        )
        report["results"][str(T)] = evaluated["results"]
        report["evaluations"][str(T)] = evaluated
        metadata["datasets"] = _dataset_metadata(manager.loaded_tasks)
        report["complete"] = len(report["evaluations"]) == len(t_values)
        _write_json(output, report)
        print(f"T={T}: {json.dumps(evaluated['results'], default=_json_default)}")
    print(f"Wrote evaluation and provenance to {output}")


if __name__ == "__main__":
    main()

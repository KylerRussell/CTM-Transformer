"""
Run lm-evaluation-harness against a trained CTM-Transformer checkpoint.

Default task suite is small-model-friendly: log-likelihood scoring on
LAMBADA, HellaSwag, PIQA, ARC-Easy. None of these need open-ended
generation, so we only have to implement `loglikelihood`.

Reference points for ~500M-param models (fully trained) so you know
what "alive" looks like:
    LAMBADA      Pythia 410M ~52% acc;  GPT-2 medium ~43%
    HellaSwag    Pythia 410M ~40%;      random 25%
    PIQA         Pythia 410M ~68%;      random 50%
    ARC-Easy     Pythia 410M ~52%;      random 25%

You're undertrained relative to those; expect lower numbers and use the
deltas over training as the signal.

Usage:

    # Single-T eval at config default
    python -m scripts.eval_harness \\
        --checkpoint checkpoints_v2_dual/latest.pt \\
        --tasks lambada_openai,hellaswag,piqa,arc_easy \\
        --device cuda:0 --batch_size 4

    # Quick smoke test on a handful of examples per task
    python -m scripts.eval_harness ... --limit 100

    # T-ablation (deliverable #4)
    python -m scripts.eval_harness ... --t_sweep 1,4,8 \\
        --output eval_t_sweep.json

The adapter handles both lm-eval 0.4.x (Instance objects) and 0.3.x
(tuple) APIs because the package's API changed at 0.4.0 and there are
both versions in the wild.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# ── lm-eval API compat ────────────────────────────────────────────
try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    _LM_EVAL_API = "0.4"
except ImportError:
    try:
        from lm_eval.base import LM  # type: ignore
        Instance = None
        _LM_EVAL_API = "0.3"
    except ImportError as e:
        raise ImportError(
            "lm-evaluation-harness not installed.\n"
            "  pip install lm-eval\n"
            f"(import failed: {e})"
        ) from e


class CTMTransformerLM(LM):
    """lm-eval-harness adapter for CTMTransformer.

    Implements `loglikelihood` only; the four tasks in our default suite
    (lambada_openai, hellaswag, piqa, arc_easy) all use likelihood
    scoring, not generation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda:0",
        batch_size: int = 4,
        max_thought_steps: Optional[int] = None,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_thought_steps = max_thought_steps
        self._max_seq_len = max_seq_len

        # Resolve EOT token id across tokenizer types
        eot = getattr(tokenizer, "eot_token", None)
        if eot is None:
            inner = getattr(tokenizer, "tokenizer", None)
            if inner is not None and getattr(inner, "eos_token_id", None) is not None:
                eot = int(inner.eos_token_id)
            else:
                eot = 0
        self._eot_token_id = int(eot)

    # ── lm-eval API surface ───────────────────────────────────────

    @property
    def eot_token_id(self):
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

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    # ── Forward helpers ──────────────────────────────────────────

    @torch.no_grad()
    def _model_call(self, ids: torch.Tensor) -> torch.Tensor:
        """Run model forward, return logits [B, S, V]."""
        result = self.model(ids, max_thought_steps=self._max_thought_steps)
        return result["logits"]

    # ── loglikelihood ────────────────────────────────────────────

    @torch.no_grad()
    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """Score each (context, continuation) pair.

        Returns: list of (log_likelihood, is_greedy) tuples in input order.
        """
        # Normalize input: lm-eval 0.4 passes Instance objects, 0.3 passes tuples
        if Instance is not None and len(requests) > 0 and isinstance(requests[0], Instance):
            req_args = [r.args for r in requests]
        else:
            req_args = list(requests)

        # ── Tokenize all pairs ────────────────────────────────────
        encoded = []
        for context, continuation in req_args:
            ctx_toks = self.tok_encode(context) if context else []
            cont_toks = self.tok_encode(continuation)
            if len(cont_toks) == 0:
                # Degenerate: no continuation. Score 0.0, mark non-greedy.
                encoded.append(([self._eot_token_id], 1, 0))
                continue
            full = ctx_toks + cont_toks

            # Truncate from the LEFT (drop oldest context) so we always
            # preserve the continuation. If the continuation alone
            # exceeds max_seq_len, truncate it from the right (rare).
            if len(full) > self._max_seq_len:
                drop = len(full) - self._max_seq_len
                if len(ctx_toks) >= drop:
                    ctx_toks = ctx_toks[drop:]
                    full = ctx_toks + cont_toks
                else:
                    cont_toks = cont_toks[: self._max_seq_len]
                    ctx_toks = []
                    full = cont_toks
            encoded.append((full, len(ctx_toks), len(cont_toks)))

        # ── Length-bucketed batching ──────────────────────────────
        order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i][0]))
        results = [None] * len(encoded)

        try:
            from tqdm.auto import tqdm
            iter_chunks = tqdm(
                range(0, len(order), self._batch_size),
                disable=disable_tqdm,
                desc=f"loglikelihood (T={self._max_thought_steps or 'cfg'})",
            )
        except ImportError:
            iter_chunks = range(0, len(order), self._batch_size)

        for batch_start in iter_chunks:
            batch_idxs = order[batch_start: batch_start + self._batch_size]
            batch = [encoded[i] for i in batch_idxs]
            max_len = max(len(b[0]) for b in batch)

            # Right-pad with EOT. We only score continuation positions,
            # which are by construction non-padding, so padding doesn't
            # contaminate the scores.
            ids = torch.full(
                (len(batch), max_len), self._eot_token_id,
                dtype=torch.long, device=self._device,
            )
            for i, (full, _, _) in enumerate(batch):
                ids[i, :len(full)] = torch.tensor(full, dtype=torch.long, device=self._device)

            logits = self._model_call(ids)  # [B, max_len, V]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            for i, (full, ctx_len, cont_len) in enumerate(batch):
                if cont_len == 0:
                    results[batch_idxs[i]] = (0.0, False)
                    continue

                # Predictions for token at position p come from logits at p-1.
                # Continuation occupies [ctx_len, ctx_len + cont_len).
                # If ctx_len == 0, we lose the first continuation token's
                # score (no logits to predict it from); same convention as
                # the standard HF lm-eval adapter.
                if ctx_len == 0:
                    pred_start = 0
                    target_start = 1
                    n_score = cont_len - 1
                else:
                    pred_start = ctx_len - 1
                    target_start = ctx_len
                    n_score = cont_len

                if n_score <= 0:
                    results[batch_idxs[i]] = (0.0, True)
                    continue

                target_ids = torch.tensor(
                    full[target_start: target_start + n_score],
                    dtype=torch.long, device=self._device,
                )
                slice_lp = log_probs[i, pred_start: pred_start + n_score]  # [n, V]
                gathered = slice_lp.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)  # [n]
                ll = float(gathered.sum().item())

                argmax_ids = slice_lp.argmax(dim=-1)
                is_greedy = bool((argmax_ids == target_ids).all().item())

                results[batch_idxs[i]] = (ll, is_greedy)

        return results

    # ── Required by API but not used by our suite ────────────────

    @torch.no_grad()
    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not implemented; not needed for "
            "lambada_openai/hellaswag/piqa/arc_easy. "
            "Implement if you add wikitext or similar PPL tasks."
        )

    def generate_until(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "generate_until not implemented; the small-model suite uses "
            "loglikelihood scoring. For GSM8K-style generation tasks, "
            "wire this through to model.generate()."
        )


# ──────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────

def _load_checkpoint(checkpoint_path, config_overrides=None):
    """Load a saved checkpoint and return (model, config)."""
    from ctm_transformer.config import CTMConfig
    from ctm_transformer.model import CTMTransformer

    print(f"  Loading state dict from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in ckpt:
        config_dict = (
            ckpt["config"] if isinstance(ckpt["config"], dict)
            else ckpt["config"].__dict__
        )
    else:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} has no 'config' field. "
            "Reconstruct CTMConfig manually and load only state_dict if needed."
        )

    if config_overrides:
        config_dict = {**config_dict, **config_overrides}

    # Filter to known fields (defensive against config schema drift)
    known = set(CTMConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in known}
    config = CTMConfig(**filtered)

    print(f"  Building model (n_layers={config.n_layers}, "
          f"d_model={config.d_model}, T={config.max_thought_steps})")
    model = CTMTransformer(config)

    # Strip wrapper prefixes from compile/DDP saves
    state = (
        ckpt.get("model")
        or ckpt.get("model_state_dict")
        or ckpt.get("state_dict")
        or ckpt
    )
    fixed = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        fixed[k] = v

    missing, unexpected = model.load_state_dict(fixed, strict=False)
    if missing:
        print(f"  [load] {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [load] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")

    return model, config


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a saved CTMTransformer checkpoint")
    parser.add_argument("--tokenizer", default=None,
                        help="Override tokenizer string; defaults to checkpoint config")
    parser.add_argument("--tasks", default="lambada_openai,hellaswag,piqa,arc_easy")
    parser.add_argument("--thought_steps", type=int, default=None,
                        help="Override max_thought_steps (single-T eval)")
    parser.add_argument("--t_sweep", default=None,
                        help="Comma-separated T values for ablation, e.g. 1,4,8. "
                             "Overrides --thought_steps if set.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples per task (for quick smoke tests)")
    parser.add_argument("--output", default=None,
                        help="Write results JSON to this path")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()

    print(f"lm-eval-harness adapter for CTM-Transformer (lm-eval API {_LM_EVAL_API})")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Tasks:       {args.tasks}")
    print(f"  Device:      {args.device}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Batch size:  {args.batch_size}")

    # ── Load model ─────────────────────────────────────────────────
    model, config = _load_checkpoint(args.checkpoint)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    model = model.to(device, dtype)
    model.eval()

    # ── Tokenizer ──────────────────────────────────────────────────
    if args.tokenizer is not None:
        config.tokenizer = args.tokenizer
    # Reuse training tokenizer factory for consistency
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ctm_transformer.train import get_tokenizer
    tokenizer = get_tokenizer(config)
    print(f"  Tokenizer:   {config.tokenizer} (vocab={tokenizer.n_vocab})")

    # ── T values ───────────────────────────────────────────────────
    if args.t_sweep:
        t_values = [int(x) for x in args.t_sweep.split(",") if x.strip()]
    elif args.thought_steps is not None:
        t_values = [args.thought_steps]
    else:
        t_values = [None]

    # ── Run ────────────────────────────────────────────────────────
    from lm_eval import simple_evaluate

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    max_seq_len = getattr(config, "max_seq_len", None) or getattr(config, "seq_len", 512)

    all_results: dict = {}
    for T in t_values:
        T_label = T if T is not None else config.max_thought_steps
        print(f"\n{'='*60}\nEvaluating at T={T_label}\n{'='*60}")

        lm = CTMTransformerLM(
            model=model, tokenizer=tokenizer,
            device=args.device, batch_size=args.batch_size,
            max_thought_steps=T, max_seq_len=max_seq_len,
        )

        eval_out = simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        per_task = eval_out.get("results", {}) if isinstance(eval_out, dict) else {}
        all_results[str(T_label)] = per_task

        for task_name, task_results in per_task.items():
            print(f"  {task_name}:")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    print(f"    {metric}: {value:.4f}")

    # ── T-sweep summary ────────────────────────────────────────────
    if len(t_values) > 1:
        print(f"\n{'='*60}\nT-ablation summary\n{'='*60}")
        first_T = next(iter(all_results.keys()))
        for task_name in all_results[first_T].keys():
            metrics = all_results[first_T][task_name]
            primary = next(
                (m for m in ("acc_norm,none", "acc,none", "acc_norm", "acc")
                 if m in metrics), None,
            )
            if primary is None:
                continue
            row = "  ".join(
                f"T={T}:{all_results[T][task_name].get(primary, float('nan')):.3f}"
                for T in all_results.keys()
            )
            print(f"  {task_name:20s} ({primary:>14s}):  {row}")

    # ── Save ───────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()

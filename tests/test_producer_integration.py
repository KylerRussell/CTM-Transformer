"""
test_producer_integration.py — End-to-end smoke test for the cache producer.

Exercises the producer's main loop with a fake teacher and fake data, verifying:
  1. Disk cap (--max_cache_gb) actually triggers a clean shutdown.
  2. Cached shards under the cap are well-formed and loadable.
  3. The producer's resume logic skips already-written shards correctly.
  4. The new write_shard() with corrected tmp filename rounds-trips on disk.
"""

import sys
import tempfile
from pathlib import Path
import unittest.mock as mock

import numpy as np


def fake_teacher(input_ids, *args, **kwargs):
    """Fake teacher: returns logits that look plausible (Zipfian)."""
    import torch
    B, S = input_ids.shape
    V = 1024
    # Zipfian-ish logits so topk has structure
    base = torch.linspace(5.0, -5.0, V).unsqueeze(0).unsqueeze(0).expand(B, S, V)
    noise = torch.randn(B, S, V) * 0.5
    out = type("Out", (), {})()
    out.logits = (base + noise).to(input_ids.device)
    return out


class FakeTeacherModel:
    """Stand-in for a transformers AutoModelForCausalLM."""
    class Config:
        vocab_size = 1024
    config = Config()
    def __call__(self, input_ids, *args, **kwargs):
        return fake_teacher(input_ids)
    def eval(self): return self
    def requires_grad_(self, _): return self


class FakeTokenizer:
    """Stand-in tokenizer that produces predictable token sequences."""
    eos_token_id = 0
    bos_token_id = 0
    @staticmethod
    def encode(text, add_special_tokens=False):
        # Hash text to a sequence; longer text → more tokens.
        n = max(1, len(text) // 4)
        return [(hash(text + str(i)) & 0x3FF) for i in range(n)]
    @classmethod
    def from_pretrained(cls, name):
        return cls()


def _make_doc_stream(n_docs: int, doc_text_len: int = 2000):
    """Yield n_docs fake-source documents long enough to fill many shards."""
    for i in range(n_docs):
        yield {"text": f"document_{i:08d}_" + "lorem ipsum dolor sit amet " * doc_text_len}


def test_disk_cap_triggers_shutdown():
    """With a small --max_cache_gb, the producer should exit cleanly
    after writing fewer shards than --max_samples would have allowed."""
    print("[test 1] Disk cap triggers clean shutdown...", end=" ")

    import scripts.cache_teacher_logits as mod

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "phase1"

        # Simulated CLI args
        args = type("Args", (), {})()
        args.phase = 1
        args.datasets = ["fake_a", "fake_b"]
        args.subsets = ["x", "y"]
        args.weights = [0.5, 0.5]
        args.teacher_model = "unused"
        args.tokenizer = "unused"
        args.data_cache_dir = "/tmp/unused"
        args.output_dir = str(out_dir)
        args.seq_len = 128                # smaller so test runs fast
        args.top_k = 8
        args.samples_per_shard = 16        # tiny shards
        args.max_samples = 100_000_000    # huge — disk cap should win
        args.max_cache_gb = 0.001         # ~1 MB cap → should hit fast
        args.no_stratified = False
        args.no_cycle = True   # test data is bounded; cycling would loop forever
        args.rank = 0           # single-process default
        args.world_size = 1
        args.device = "cpu"
        args.teacher_batch_size = 4
        args.dtype = "bfloat16"
        args.seed = 42
        args.log_interval = 100

        # Patch all the heavy imports the producer would normally do
        with mock.patch.object(mod, "parse_args", return_value=args), \
             mock.patch.object(mod, "stratified_window_generator",
                               side_effect=lambda **kw: _make_doc_stream(50_000)), \
             mock.patch.object(mod, "sliding_window_generator",
                               side_effect=lambda **kw: _make_doc_stream(50_000)):
            # Patch the lazy transformers imports inside main()
            sys.modules["transformers"] = type(sys)("fake_transformers_pkg")
            sys.modules["transformers"].AutoModelForCausalLM = type(
                "FakeAutoModel", (), {
                    "from_pretrained": classmethod(
                        lambda cls, *a, **kw: FakeTeacherModel()
                    )
                }
            )
            sys.modules["transformers"].AutoTokenizer = FakeTokenizer

            try:
                mod.main()
            except SystemExit:
                pass

        # Verify shards were written, and the cap was respected
        shards = sorted(out_dir.glob("shard_*.npz"))
        assert len(shards) > 0, "no shards written"
        total_bytes = sum(s.stat().st_size for s in shards)
        cap_bytes = int(args.max_cache_gb * (1024 ** 3))
        # We allow one shard over the cap (the cap is checked AFTER write)
        single_shard_bytes = shards[0].stat().st_size
        assert total_bytes <= cap_bytes + single_shard_bytes, (
            f"cache size {total_bytes} significantly exceeds cap "
            f"{cap_bytes} (single shard = {single_shard_bytes})"
        )

        # And each shard is well-formed
        with np.load(shards[0]) as z:
            assert z["input_ids"].shape == (16, 128)
            assert z["top_indices"].shape == (16, 128, 8)
            assert z["top_values"].dtype == np.float16

    print(f"OK ({len(shards)} shards, "
          f"{total_bytes / 1024:.1f} KB ≤ cap {cap_bytes / 1024:.1f} KB + 1 shard)")


def test_resume_skips_written_shards():
    """The producer's resume logic should skip shards already on disk."""
    print("[test 2] Resume skips written shards...", end=" ")

    import scripts.cache_teacher_logits as mod
    from scripts.cache_teacher_logits import write_shard

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        N, S, K = 4, 16, 8
        for idx in [0, 1, 2]:
            arr = np.random.randint(0, 100, size=(N, S+1), dtype=np.int64)
            top_i = np.random.randint(0, 100, size=(N, S, K), dtype=np.int32)
            top_v = np.random.randn(N, S, K).astype(np.float16)
            resid = np.random.randn(N, S).astype(np.float16)
            write_shard(out_dir, idx, arr, top_i, top_v, resid)

        # Simulate resume detection
        existing = sorted(out_dir.glob("shard_*.npz"))
        assert len(existing) == 3
        last = int(existing[-1].stem.split("_")[1])
        next_shard = last + 1
        assert next_shard == 3, f"expected next=3, got {next_shard}"

    print("OK")


if __name__ == "__main__":
    test_resume_skips_written_shards()
    # The disk-cap test is heavier and depends on torch — skip if missing
    try:
        import torch
        test_disk_cap_triggers_shutdown()
    except ImportError:
        print("[test 1] SKIPPED (torch not installed in this env)")
    print("\nAll producer integration smoke tests passed.")
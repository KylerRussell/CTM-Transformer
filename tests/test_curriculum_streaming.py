"""
test_curriculum_streaming.py — Smoke test for the producer's
curriculum-aware text streamer.

Mocks sliding_window_generator so we can verify the weighted-interleave
logic without hitting HuggingFace. Checks that:
  1. The streamer respects per-source weights (statistical test).
  2. Exhausted sources are removed from the active pool.
  3. Resume-skip alignment gives identical samples across runs.

Run:
    PYTHONPATH=. python test_curriculum_streaming.py
"""

import sys
import unittest.mock as mock
from collections import Counter


def fake_generator(name: str, n: int):
    """Yields {'text': f'{name}_<i>'} n times, then stops."""
    def _gen(*args, **kwargs):
        for i in range(n):
            yield {"text": f"{name}_{i}"}
    return _gen


def _patch_both_generators(mod, side_effect):
    """Helper: patch BOTH the sequential and stratified generators with
    the same fake factory. The stream_curriculum_text default is
    stratified=True; we patch both so tests don't depend on which
    backend is selected."""
    p1 = mock.patch.object(mod, "sliding_window_generator", side_effect=side_effect)
    p2 = mock.patch.object(mod, "stratified_window_generator", side_effect=side_effect)
    return p1, p2


def test_weighted_interleave():
    """With weights [0.7, 0.3] and abundant source data, the cached output
    should reflect the weights to within a few percent."""
    print("[test 1] Weighted interleave matches mixing weights...", end=" ")

    import scripts.cache_teacher_logits as mod

    factory = lambda ds_name, subset, **kw: (
        ({"text": f"{ds_name}_{i}"} for i in range(10000))
    )
    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.7, 0.3],
            data_cache_dir="/tmp/unused",
            seed=42,
        )
        counts = Counter()
        for i, doc in enumerate(gen):
            src = doc["text"].split("_")[0]
            counts[src] += 1
            if i >= 5000:
                break

        ratio_alpha = counts["alpha"] / (counts["alpha"] + counts["beta"])
        assert 0.67 <= ratio_alpha <= 0.73, (
            f"alpha share = {ratio_alpha:.3f}, expected ~0.7 "
            f"(counts: {dict(counts)})"
        )
    print(f"OK (alpha={ratio_alpha:.3f})")


def test_source_exhaustion():
    """When one source runs out, the streamer should switch to the others.

    NOTE: this test uses cycle_exhausted=False to preserve the legacy
    drop-on-exhaustion semantics. With cycling on (the new default),
    the small source would cycle indefinitely — see
    test_cycle_exhausted_preserves_weights for the cycling case."""
    print("[test 2] Exhausted sources fall out of the pool...", end=" ")

    import scripts.cache_teacher_logits as mod

    def factory(ds_name, subset, **kw):
        n = 5 if ds_name == "alpha" else 1000
        return ({"text": f"{ds_name}_{i}"} for i in range(n))

    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused",
            seed=0,
            cycle_exhausted=False,
        )
        counts = Counter()
        for doc in gen:
            src = doc["text"].split("_")[0]
            counts[src] += 1
        assert counts["alpha"] == 5, f"alpha count={counts['alpha']}, expected 5"
        assert counts["beta"] == 1000, f"beta count={counts['beta']}, expected 1000"
    print(f"OK (alpha=5, beta=1000)")


def test_resume_alignment():
    """Same seed → same sample sequence. Critical for resume correctness."""
    print("[test 3] Resume alignment is deterministic...", end=" ")

    import scripts.cache_teacher_logits as mod

    def factory(ds_name, subset, **kw):
        return ({"text": f"{ds_name}_{i}"} for i in range(1000))

    samples_run_a, samples_run_b = [], []
    for target in (samples_run_a, samples_run_b):
        p1, p2 = _patch_both_generators(mod, factory)
        with p1, p2:
            gen = mod.stream_curriculum_text(
                datasets=["alpha", "beta", "gamma"],
                subsets=["s1", "s2", "s3"],
                weights=[0.5, 0.3, 0.2],
                data_cache_dir="/tmp/unused",
                seed=12345,
            )
            for i, doc in enumerate(gen):
                target.append(doc["text"])
                if i >= 200:
                    break

    assert samples_run_a == samples_run_b, (
        f"Runs diverged: first diff at index "
        f"{next(i for i, (a, b) in enumerate(zip(samples_run_a, samples_run_b)) if a != b)}"
    )
    print(f"OK ({len(samples_run_a)} samples identical)")


def test_cycle_exhausted_preserves_weights():
    """With cycle_exhausted=True, a small source that runs out gets
    restarted, so the long-run mixing ratio still reflects the configured
    weights. With cycle_exhausted=False, it just disappears."""
    print("[test 5] cycle_exhausted preserves long-run mixing ratios...", end=" ")

    import scripts.cache_teacher_logits as mod

    # alpha is small (50 docs), beta is huge (10000 docs)
    def factory(ds_name, subset, **kw):
        n = 50 if ds_name == "alpha" else 10000
        return ({"text": f"{ds_name}_{i}"} for i in range(n))

    # ── With cycling: ratio should stay near 0.5/0.5 over a long run ──
    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused",
            seed=0,
            cycle_exhausted=True,
        )
        counts = Counter()
        for i, doc in enumerate(gen):
            counts[doc["text"].split("_")[0]] += 1
            if i >= 2000:
                break
        ratio_alpha_cycle = counts["alpha"] / (counts["alpha"] + counts["beta"])
        # With cycling, alpha shows up ~50% of the time despite only
        # having 50 unique docs (it gets cycled ~20+ times in 2000 samples).
        assert 0.45 <= ratio_alpha_cycle <= 0.55, (
            f"cycling: alpha share = {ratio_alpha_cycle:.3f}, expected ~0.5"
        )

    # ── Without cycling: alpha runs out at 50, beta dominates the rest ──
    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused",
            seed=0,
            cycle_exhausted=False,
        )
        counts = Counter()
        for i, doc in enumerate(gen):
            counts[doc["text"].split("_")[0]] += 1
            if i >= 2000:
                break
        # alpha capped at 50 unique docs; beta provides the rest (~1950)
        assert counts["alpha"] == 50, (
            f"no-cycle: alpha count={counts['alpha']}, expected 50"
        )
        assert counts["beta"] >= 1900, (
            f"no-cycle: beta count={counts['beta']}, expected ~1950"
        )

    print(f"OK (cycle: alpha={ratio_alpha_cycle:.2f}, no-cycle: alpha=50)")


def test_cycle_drops_truly_empty_source():
    """If a source has zero readable data, cycling should drop it after
    one re-attempt to avoid an infinite loop. The non-empty source(s)
    should keep providing data (cycling indefinitely)."""
    print("[test 6] cycle_exhausted drops truly-empty sources...", end=" ")

    import scripts.cache_teacher_logits as mod

    def factory(ds_name, subset, **kw):
        # alpha is completely empty; beta has data
        if ds_name == "alpha":
            return iter([])
        return ({"text": f"{ds_name}_{i}"} for i in range(100))

    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused",
            seed=0,
            cycle_exhausted=True,
        )
        counts = Counter()
        # Cap iteration so beta-cycling doesn't run forever.
        for i, doc in enumerate(gen):
            counts[doc["text"].split("_")[0]] += 1
            if i >= 500:
                break
        # alpha should never appear (dropped after empty re-attempt);
        # beta provides everything within the iteration cap.
        assert counts.get("alpha", 0) == 0, (
            f"alpha had no data; got count={counts.get('alpha', 0)}"
        )
        # All 501 picks should have gone to beta (cycling its 100 docs).
        assert counts["beta"] == 501, (
            f"beta should have provided 501; got {counts['beta']}"
        )

    print("OK")


def test_cycle_small_source_keeps_appearing():
    """Regression test for the bug seen in production: small sources
    (1-2 parquet files) were being dropped on cycle 2 because the
    eval-tail reservation was eating their content. After the fix,
    a small source should stay in the cycle pool and keep yielding
    its content multiple times during a long run."""
    print("[test 7] Small sources keep cycling (regression)...", end=" ")

    import scripts.cache_teacher_logits as mod

    # alpha is tiny (10 docs), beta is normal (500 docs)
    def factory(ds_name, subset, **kw):
        n = 10 if ds_name == "alpha" else 500
        return ({"text": f"{ds_name}_{i}"} for i in range(n))

    p1, p2 = _patch_both_generators(mod, factory)
    with p1, p2:
        gen = mod.stream_curriculum_text(
            datasets=["alpha", "beta"],
            subsets=["s1", "s2"],
            weights=[0.3, 0.7],   # alpha gets 30% of picks
            data_cache_dir="/tmp/unused",
            seed=42,
            cycle_exhausted=True,
        )
        counts = Counter()
        # Pull 1000 docs total. With weight 0.3, alpha should get ~300 picks,
        # which means ~30 cycles through its 10-doc data. With weight 0.7,
        # beta gets ~700 picks, ~1.4 cycles through its 500-doc data.
        for i, doc in enumerate(gen):
            counts[doc["text"].split("_")[0]] += 1
            if i >= 1000:
                break

        # The key assertion: alpha must appear MANY times, not just 10
        # (its raw data size). If alpha appears only ~10 times, the
        # dropping-on-cycle-2 bug is back.
        assert counts["alpha"] >= 250, (
            f"alpha cycled too few times: {counts['alpha']} (expected ~300). "
            f"This is the bug where small sources get dropped on cycle 2."
        )
        # Sanity: the ratio should be close to the weights.
        ratio_alpha = counts["alpha"] / (counts["alpha"] + counts["beta"])
        assert 0.27 <= ratio_alpha <= 0.33, (
            f"alpha ratio = {ratio_alpha:.3f}, expected ~0.30"
        )

    print(f"OK (alpha cycled to {counts['alpha']} appearances "
          f"from 10 unique docs, ratio={ratio_alpha:.2f})")


def test_stratified_flag_dispatches():
    """The stratified flag should select the correct underlying generator.
    With stratified=False, only sliding_window_generator should be called.
    With stratified=True, only stratified_window_generator should be called."""
    print("[test 4] stratified flag dispatches to correct backend...", end=" ")

    import scripts.cache_teacher_logits as mod

    factory = lambda ds_name, subset, **kw: (
        ({"text": f"{ds_name}_{i}"} for i in range(50))
    )

    # stratified=False → sliding called, stratified NOT called
    seq_calls, strat_calls = [], []
    with mock.patch.object(mod, "sliding_window_generator",
                           side_effect=lambda **kw: (seq_calls.append(1), factory(**kw))[1]), \
         mock.patch.object(mod, "stratified_window_generator",
                           side_effect=lambda **kw: (strat_calls.append(1), factory(**kw))[1]):
        list(mod.stream_curriculum_text(
            datasets=["a", "b"], subsets=["x", "y"], weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused", seed=0, stratified=False,
            cycle_exhausted=False,  # so list() actually terminates
        ))
    assert len(seq_calls) == 2 and len(strat_calls) == 0, (
        f"stratified=False: seq={len(seq_calls)} strat={len(strat_calls)}"
    )

    # stratified=True → stratified called, sliding NOT called
    seq_calls, strat_calls = [], []
    with mock.patch.object(mod, "sliding_window_generator",
                           side_effect=lambda **kw: (seq_calls.append(1), factory(**kw))[1]), \
         mock.patch.object(mod, "stratified_window_generator",
                           side_effect=lambda **kw: (strat_calls.append(1), factory(**kw))[1]):
        list(mod.stream_curriculum_text(
            datasets=["a", "b"], subsets=["x", "y"], weights=[0.5, 0.5],
            data_cache_dir="/tmp/unused", seed=0, stratified=True,
            cycle_exhausted=False,  # so list() actually terminates
        ))
    assert len(seq_calls) == 0 and len(strat_calls) == 2, (
        f"stratified=True: seq={len(seq_calls)} strat={len(strat_calls)}"
    )
    print("OK")


if __name__ == "__main__":
    test_weighted_interleave()
    test_source_exhaustion()
    test_resume_alignment()
    test_cycle_exhausted_preserves_weights()
    test_cycle_drops_truly_empty_source()
    test_cycle_small_source_keeps_appearing()
    test_stratified_flag_dispatches()
    print("\nAll curriculum-streaming smoke tests passed.")
"""
phase_timer.py — drop-in utility for measuring where time is spent in
the training loop. Usage:

    from phase_timer import PhaseTimer
    timer = PhaseTimer(enabled=True, warmup_steps=10)

    for step in range(...):
        with timer.step():
            with timer("data"):
                batch = next(loader)
            with timer("teacher"):
                teacher_logits = teacher(...)
            with timer("student_fwd"):
                out = model(...)
            with timer("loss"):
                loss = compute_loss(out, ...)
            with timer("backward"):
                loss.backward()
            with timer("optimizer"):
                opt.step(); opt.zero_grad()

        if step % 50 == 0 and step > 0:
            timer.report()

The timer issues torch.cuda.synchronize() at every boundary, so the
numbers reflect actual elapsed wall time on the GPU — not just kernel
launch time. Skip the first few steps with `warmup_steps` so initial
allocator/graph-capture costs don't pollute the numbers.
"""
import time
import statistics
from contextlib import contextmanager
from collections import defaultdict

import torch


class PhaseTimer:
    def __init__(self, enabled: bool = True, warmup_steps: int = 5):
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.timings: dict[str, list[float]] = defaultdict(list)
        self.step_count = 0
        self._is_cuda = torch.cuda.is_available()
        self._in_warmup = True

    def _sync(self):
        if self._is_cuda:
            torch.cuda.synchronize()

    @contextmanager
    def step(self):
        """Wrap each training step. Increments step counter and toggles
        warmup state so the first few steps don't dominate the stats."""
        if not self.enabled:
            yield
            return
        self.step_count += 1
        self._in_warmup = self.step_count <= self.warmup_steps
        yield

    @contextmanager
    def __call__(self, name: str):
        """Time a phase. No-ops if timer is disabled or in warmup."""
        if not self.enabled or self._in_warmup:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.timings[name].append(time.perf_counter() - t0)

    def report(self, last_n: int | None = None, sort_by: str = "mean"):
        """Print a summary table. last_n=N restricts to the last N samples;
        sort_by ∈ {'mean', 'median', 'name'}."""
        if not self.timings:
            print("[PhaseTimer] no samples yet")
            return

        steps_recorded = max(1, self.step_count - self.warmup_steps)
        rows = []
        total_mean_per_step = 0.0
        for name, samples in self.timings.items():
            true_n = len(self.timings[name])
            if last_n is not None:
                samples = samples[-last_n:]
            n = len(samples)
            if n == 0:
                continue
            mean_ms = statistics.mean(samples) * 1000
            med_ms = statistics.median(samples) * 1000
            sorted_s = sorted(samples)
            p95_ms = sorted_s[min(int(n * 0.95), n - 1)] * 1000
            
            mean_per_step = mean_ms * (true_n / steps_recorded)
            rows.append((name, mean_ms, med_ms, p95_ms, n, mean_per_step))
            total_mean_per_step += mean_per_step

        if sort_by == "mean":
            rows.sort(key=lambda r: -r[1])
        elif sort_by == "median":
            rows.sort(key=lambda r: -r[2])
        else:
            rows.sort(key=lambda r: r[0])

        bar = "─" * 72
        print()
        print(bar)
        print(f"PhaseTimer  (step {self.step_count}, warmup {self.warmup_steps})")
        print(bar)
        print(f"{'phase':<24s} {'mean(ms)':>10s} {'median':>9s} {'p95':>9s} "
              f"{'n':>5s} {'%total':>8s}")
        print(bar)
        for name, mean_ms, med_ms, p95_ms, n, mean_per_step in rows:
            pct = 100 * mean_per_step / total_mean_per_step if total_mean_per_step > 0 else 0
            print(f"{name:<24s} {mean_ms:10.2f} {med_ms:9.2f} {p95_ms:9.2f} "
                  f"{n:5d} {pct:7.1f}%")
        print(bar)
        print(f"{'TOTAL (mean per step)':<24s} {total_mean_per_step:10.2f} ms")
        print(bar)
        print()

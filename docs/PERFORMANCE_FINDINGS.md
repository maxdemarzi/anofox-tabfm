# anofox-tabfm — performance findings (2026-08-13)

Measured on the `performance` branch off `503bc59`. Box: 12-core Linux, idle
(load < 0.6), `reldebug` cpu flavor, prebuilt ORT 1.23.2, `perf` at
`perf_event_paranoid=-1`. Harness: `test/benchmarking/benchmark_predict.py`.

Method follows swan's `docs/dev/PERFORMANCE_TUNING.md` — paired arms alternating
adjacent in time, ratio of medians, both arms asserted to return the same answer.
Complements `docs/PERFORMANCE_PLAN.md`, which measures the **real** models; this
document measures the **plumbing** and is explicit about the difference.

## What the fixture does and does not license

Every number here drives the committed random-init fixture model, the only
weight-free model that runs locally (license wall, working rule 6). The fixture's
forward pass is far cheaper than a real checkpoint, so:

- **Transferable:** the absolute cost of anything *outside* the forward pass.
  Preprocessing 100k cells costs what it costs regardless of the model behind it.
- **Not transferable:** any *percentage* split between our code and ORT, and any
  thread-scaling curve. `PERFORMANCE_PLAN.md` measures the real 6.6 GB model at
  `CPU ≈ 0.35 s + 0.010 s × T` — a different regime entirely (bandwidth-bound at
  small T, compute-bound at large T).

## Workloads

Two shapes, because they scale on different axes and nothing here separated them
before. Both are in the committed harness.

- **wide** — 60 train / 40 test rows, N feature columns, feature 0 carrying the
  signal and the rest noise. Ported from duckdb-rocket's
  `scripts/probe_anofox.py:synthetic_data`; its probes recorded 500 features at
  6.8 s and 1000 at 17.2 s against the real `tabicl-v2`.
- **tall** — 8 features, many rows: the batch axis, which `wide` never touches.

Baseline, fixture model, single query:

| shape | time |
|---|---:|
| wide, 100 features | 78 ms |
| wide, 500 features | 507 ms |
| wide, 1000 features | 1826 ms |
| tall, 4000 rows | 256 ms |

Feature count is superlinear (~2× features → ~3.6× time), consistent with
attention over the feature axis.

## Declined: the boxed-`Value` ingest and preprocess path

Reading the code found three textbook instances of `PERFORMANCE_TUNING.md` §7
(typed extraction instead of boxed values) and §5 (allocate once per worker):

1. `tabfm_predict_agg.cpp:357-362` — every row is boxed into a `Value`, written
   into a **one-row `DataChunk`**, and appended to a `ColumnDataCollection`
   individually, though the aggregate hands us `count` rows at a time.
2. `tabfm_preprocess.cpp:205` — `data.GetRows()`, then a boxed `Value` per cell
   in every fit and transform loop, a heap `std::string` per categorical cell,
   and `std::map`/`std::set` (red-black trees, string-keyed) for the category
   fit.
3. `tabfm_predict_agg.cpp:545-552` — every row materialized into a
   `vector<Value>`, then again into a `vector<vector<Value>>`.

**All three declined, unmeasured-as-wins.** `perf record` on the 1000-feature
workload, by DSO:

```
80.52%  libonnxruntime.so.1.23.2
13.29%  [unknown]  (kernel)
 0.03%  duckdb::anofox::PreprocessBatch(...)
```

`PreprocessBatch` is **0.03%** of the profile. Rewriting all three could not
return more than a rounding error on this path. This is §9's rule landing a
fourth time: the shape was right and the time was not there. Recorded rather than
deleted, because the *reasoning* stays useful — if a future change makes the
context path dominate (see the fork's issue #1 on context re-encoding), these are
where to start, and the profile is the thing that will say so.

## The thread default is above its own optimum — on the fixture

`anofox_tabfm_threads` defaults to `cores/2`. Sweeping it, wall-clock against
CPU burned:

| threads | wide 1000: real / user | tall 4000: real / user |
|---:|---:|---:|
| 1 | 2.240 s / 2.37 s | 0.328 s / 0.62 s |
| 2 | 1.922 s / 4.60 s | 0.260 s / 0.58 s |
| 4 | **1.717 s** / 4.10 s | **0.233 s** / 0.42 s |
| 6 | 1.789 s / 5.63 s | 0.244 s / 0.78 s |
| 8 | 1.707 s / 7.00 s | 0.251 s / 0.94 s |
| 12 | 1.742 s / 10.24 s | 0.261 s / 1.13 s |

Both shapes bottom out at **4 threads** and get *worse* beyond it, while CPU
grows roughly linearly. 1 → 12 threads buys 22% wall-clock for 4.3× the CPU.

**This is a hypothesis about the real models, not a finding about them.** The
fixture's ops are small enough that thread sync dominates them; a real checkpoint
has far larger matmuls and may well scale past 4. What makes it worth chasing is
that `PERFORMANCE_PLAN.md` independently puts the real model in a
*bandwidth-bound* regime at small T — and bandwidth-bound work does not scale
with threads either. On a 64-core pod the default resolves to 32
(post-`container-aware-thread-default`; 128 before it), and if the real curve
saturates anywhere near the fixture's, that is 8× more threads than help.

**To settle it:** run this sweep on the pod against `tabicl-v2`. duckdb-rocket's
`scripts/probe_anofox.py` already generates the workload.

## Taken: stop the intra-op pool spinning

ORT's intra-op thread pool busy-waits between ops by default — CPU traded for
wake-up latency. That is a good trade for a process that owns the machine and a
bad one here, because DuckDB runs one session per concurrent task, so every
task's spinners contend with every other task's working threads.
`session.intra_op.allow_spinning=0` in `tabfm_ort_engine.cpp`. Answers identical
in every arm:

| case | before | after | ratio |
|---|---:|---:|---:|
| tall 4000 rows, wall | 241 ms | 229 ms | **1.05×** |
| tall 4000 rows, **CPU** | 1.07 s | 0.52 s | **2.07×** |
| wide 1000 features, wall | 1718 ms | 1731 ms | 0.99× (noise) |
| 6 concurrent × 6 threads, makespan | 18.10 s | 17.22 s | **1.05×** |
| 6 concurrent × 6 threads, **CPU** | 93.3 s | 85.8 s | **1.09×** |

The wall-clock win is small and one row is a wash. The CPU reduction is the
point: on an idle box a spinning thread and a working thread are
indistinguishable, because both occupy a core nothing else wants. On a loaded pod
they are not, and load is the reported symptom.

Caveat: this trades away wake-up latency, so a workload of many tiny
back-to-back `Run()` calls could regress. Neither shape here did.

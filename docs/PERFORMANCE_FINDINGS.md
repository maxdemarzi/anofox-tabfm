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

---

# Round 2 — the cold path

The above profiles the **warm** path only. `PERFORMANCE_TUNING.md` says to report
COLD and WARM separately, and the two turn out to be entirely different queries.

## Cold is 30x warm, before any model is big

Three identical `tabfm_classify` calls in one process, 103 KB fixture:

| call | time |
|---|---:|
| 1st (loads the model) | 122 ms |
| 2nd | 6 ms |
| 3rd | 4 ms |

The load is **30x** a warm call, and that is against a 103 KB fixture. The real
`tabicl-v2` is 110 MB and `PERFORMANCE_PLAN.md`'s reference model is 6.6 GB.

Profiling this needs care: 25 fresh processes attribute almost everything to
process startup and `libonnxruntime` does not even appear. Registering the same
fixture under 40 distinct ids forces 40 real loads inside **one** process (~97 ms
each, reproducible), which is what the numbers below profile.

## The load path is 57% DuckDB scheduler and 0.06% us

```
27.4%  libonnxruntime.so.1.23.2        (session creation — real work)
57.2%  DuckDB executor + mutex + malloc
       pthread_mutex_lock 10.1%, PendingQueryResult::CheckExecutableInternal 7.4%,
       Executor::ExecuteTask 6.8%, PendingQueryResult::ExecuteInternal 6.5%,
       _int_free 4.6%, _int_malloc 3.0%, malloc 2.9%, ...
 0.06% duckdb::anofox::*  (ParseModelSpec, LoadOrGetSession, ResolveModel)
```

That 57% is DuckDB's worker threads polling while one task blocks. `Predict()`
loads the model inside the aggregate's **finalize** — a task — under the device
mutex (`tabfm_engine.cpp:871`), so for the whole load every other worker spins.

Varying DuckDB's own `SET threads` over 40 loads:

| `SET threads` | total real | total user |
|---:|---:|---:|
| 1 | 3.77 s | 3.68 s |
| 2 | 3.77 s | 3.90 s |
| 4 | 3.86 s | 6.78 s |
| 8 | 3.89 s | 6.83 s |

**Wall-clock is flat; CPU nearly doubles.** ~77 ms of CPU is burned per load for
no wall-clock gain.

**Control** (per "attribute a failure before explaining it"): 40 **warm** queries,
same sweep, cost 0.15 s → 0.25 s of CPU — 2.5 ms per query of baseline
multi-thread overhead, against 77 ms per load. The extra CPU tracks the duration
of the blocking load, not the query count. The attribution holds.

This is the same shape as the ORT spinner, one level up, and it is DuckDB's
scheduler rather than our code. The lever we own is *not blocking a task thread
for the length of a model load* — which is a real architectural change (load at
bind, or asynchronously), justified by CPU rather than wall-clock. Not attempted
here.

## `tabfm_load` cannot preload a registered model

The obvious mitigation — take the load off the query path — already has a
surface: `CALL tabfm_load('classification', model := 'x')`. It rejects a model
registered with `tabfm_register_model(base_dir := ...)`, because it checks
whether the weights were *downloaded*. So the mitigation is unavailable for
exactly the local models the fixtures use. Its error message also prints the
**task** where it names the model: asking for `model := 'fixture'` reports
`model 'classification' is not downloaded` (`tabfm_weights.cpp:705`).

## Measured and not a bottleneck

`tabfm_generate` (1500 rows from a 1500-row source, tabpfn fixture): **76 ms**.
`tabfm_impute` (1500 rows, ~20% NULLs across 3 columns): **44 ms**. Neither
shows a hot spot worth chasing at this scale.

## What cannot be measured on this box

The largest committed fixture weight file is 929 KB. `tabicl-v2` is 110 MB and
the `PERFORMANCE_PLAN.md` reference model is 6.6 GB — 100x to 7000x more bytes
through the safetensors parse, the BF16→F32 upcast and the initializer
injection. At fixture scale that whole path is 0.06% of the profile, which
licenses **no** conclusion about it at real scale. Anyone with a pod and real
weights should re-run the 40-load profile there before assuming the parse is
free; it is the one part of the cold path whose cost is genuinely proportional
to model size.

---

# Round 3 — real weights

`~/.cache/anofox-tabfm/jingang__TabICL@main` already held real **tabicl-v2**
weights (105 MB classification, 109 MB regression, BSD-3, no token needed), so
the two questions rounds 1 and 2 had to leave open are now answered on a real
model rather than the fixture.

Cold 330 ms / warm 30 ms, against the fixture's 122 ms / 4 ms. Note the cold path
grew only 2.7x for 1000x the weight bytes — so the safetensors/ckpt parse is
**not** proportional-dominant, which is the opposite of what round 2 warned might
be true. Session creation, not byte-shovelling, is the cold cost.

## The thread saturation transfers — it was not a fixture artefact

Round 1 recorded "both shapes bottom out at 4 threads" as an explicit
*hypothesis*, on the grounds that the fixture's ops might be too small to
parallelise. It reproduces on real weights. Sweeping 1..12 on a 12-core box, so
no setting is oversubscribed, real / user:

| threads | 500 features x 100 rows | 3000 rows x 8 features |
|---:|---|---|
| 1 | 2.343 s / 4.37 s | 1.867 s / 3.65 s |
| 2 | 1.733 s / 4.50 s | 1.206 s / 3.32 s |
| 4 | **1.412 s** / 5.50 s | 0.961 s / 3.98 s |
| 8 | 1.387 s / 8.79 s | **0.908 s** / 5.96 s |
| 12 | 1.438 s / 11.47 s | 0.920 s / 7.24 s |

Useful parallelism ends at 4-8. Past that nothing gets faster and CPU keeps
climbing — it is a property of the graph, not of the machine.

Forcing the counts a large host would pick for itself:

| threads | 500 features x 100 rows | 3000 rows x 8 features |
|---:|---|---|
| 8 | 1.398 s / 7.49 s | 0.889 s / 5.78 s |
| 16 | 1.462 s / 11.55 s | 0.907 s / 7.23 s |
| 32 | 1.571 s / 12.23 s | 0.895 s / 7.29 s |
| 64 | 1.911 s / 12.69 s | 0.916 s / 7.05 s |

A 64-core host defaulting to 32 is **12% slower for 63% more CPU** than 8 on the
wide shape. Read that second table as indicative only — 32 and 64 threads on a
12-core box are oversubscribed at the hardware level in a way they would not be
on a 64-core pod. The 1..12 sweep is the clean evidence, and it already shows
saturation by 8.

**Taken:** `anofox_tabfm_threads` now defaults to `min(cores/2, 8)`
(`tabfm_settings.cpp`). Hosts with ≤16 cores are unaffected — this box still
defaults to 6 — and the setting remains settable for a model or batch that
scales further. It composes with `container-aware-thread-default`: that branch
fixes *which* core count is counted, this caps what the count is allowed to
produce. A 64-core pod goes 128 → 32 (that branch) → 8 (this one).

## The spinning change holds on real weights

Re-run of round 1's change against tabicl-v2, default threads:

| shape | wall | CPU |
|---|---|---|
| 500 features x 100 rows | 1.460 → 1.491 s (0.98x) | 7.73 → 6.20 s (**1.25x**) |
| 3000 rows x 8 features | 1.012 → 0.967 s (**1.05x**) | 5.81 → 5.17 s (**1.12x**) |

Same conclusion as on the fixture: wall-clock neutral to slightly better, CPU
down 12-25%.

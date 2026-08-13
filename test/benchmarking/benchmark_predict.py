#!/usr/bin/env python3
"""Paired A/B benchmark for the tabfm_classify path.

Method follows swan's docs/dev/PERFORMANCE_TUNING.md "Measuring honestly": arms
alternate adjacent in time and the report is the ratio of medians, so ambient
load lands on both arms equally; both arms must return the same answer or the
run fails, so an A/B can never silently compare a correct build against a broken
one.

Two workload shapes, because they scale on different axes and the codebase had
no benchmark that separated them:

  wide -- 60 train / 40 test rows, N feature columns, feature 0 carrying the
          signal and the rest noise. Ported from duckdb-rocket's
          scripts/probe_anofox.py:synthetic_data, which is the ROCKET situation
          in miniature; its probes recorded 500 features at 6.8 s and 1000 at
          17.2 s against the real tabicl-v2.
  tall -- few features, many rows: the batch axis, which `wide` never exercises.

  --contend K runs K copies at once and reports makespan and total CPU. A single
  query on an idle box cannot distinguish a thread that is working from one that
  is spinning; contention can, and contention is what a pod actually runs.

All shapes drive the committed random-init fixture model, so they measure the
PLUMBING, not the model. See docs/PERFORMANCE_FINDINGS.md for what that does and
does not license you to conclude.

  ./test/benchmarking/benchmark_predict.py --arm base=build/reldebug/duckdb \
      --shape wide --features 1000 --rounds 5
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

RUNTIME_RE = re.compile(r"Run Time \(s\): real ([0-9.]+) user ([0-9.]+) sys ([0-9.]+)")

REGISTER = """
CALL tabfm_register_model(
  id := 'fixture', base_dir := 'test/fixtures',
  classification_graph := 'graph_fixture.onnx',
  classification_weights := 'model.safetensors',
  classification_tensor_map := 'tensor_map_fixture.json',
  license := 'fixture-mit', preprocessing_profile := 'tabfm_v1_minimal');
SET anofox_tabfm_max_features = 4000;
"""


def wide_sql(n_features, n_train=60, n_test=40, cats=0):
    cols = []
    for j in range(n_features):
        noise = f"(hash(i * 1000 + {j}) % 1000) / 1000.0"
        if j == 0:
            cols.append(f"CASE WHEN i % 2 = 0 THEN 1.0 ELSE -1.0 END + 0.25 * {noise} AS f{j}")
        elif j <= cats:
            cols.append(f"('c' || (hash(i * 1000 + {j}) % 7))::VARCHAR AS f{j}")
        else:
            cols.append(f"{noise} AS f{j}")
    return f"""
CREATE OR REPLACE TABLE base AS
SELECT i AS id,
       CASE WHEN i < {n_train} THEN (i % 2)::VARCHAR ELSE NULL END AS y,
       {",".join(chr(10) + "       " + c for c in cols)}
FROM range({n_train + n_test}) t(i);
"""


def tall_sql(n_rows, n_features=8, cats=2):
    cols = []
    for j in range(n_features):
        noise = f"(hash(i * 1000 + {j}) % 1000) / 1000.0"
        cols.append(f"('c' || (hash(i * 1000 + {j}) % 7))::VARCHAR AS f{j}"
                    if j < cats else f"{noise} AS f{j}")
    train = int(n_rows * 0.6)
    return f"""
SET anofox_tabfm_max_rows = {n_rows * 2};
CREATE OR REPLACE TABLE base AS
SELECT i AS id,
       CASE WHEN i < {train} THEN (i % 3)::VARCHAR ELSE NULL END AS y,
       {",".join(chr(10) + "       " + c for c in cols)}
FROM range({n_rows}) t(i);
"""


def build_script(setup, features, threads):
    flist = "[" + ", ".join(f"'f{j}'" for j in range(features)) + "]"
    query = f"SELECT * FROM tabfm_classify('base', 'y', model := 'fixture', features := {flist})"
    thread_set = f"SET anofox_tabfm_threads = {threads};\n" if threads else ""
    return f"""{REGISTER}{thread_set}{setup}
-- warm: the first call pays model load + ORT session creation
CREATE OR REPLACE TEMP TABLE warm AS SELECT count(*) c FROM ({query});
.timer on
CREATE OR REPLACE TEMP TABLE r AS {query};
.timer off
SELECT count(*) AS n, round(sum(yhat_score)::DECIMAL(18,4), 4) AS s,
       sum(hash(yhat)) AS h FROM r;
"""


def write_sql(sql):
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        return f.name


def run_once(binary, sql_path, cwd):
    env = dict(os.environ, DATAZOO_DISABLE_TELEMETRY="1")
    out = subprocess.run([binary, "-unsigned", "-noheader", "-list", "-f", sql_path],
                         capture_output=True, text=True, cwd=cwd, env=env, timeout=3600)
    if out.returncode != 0:
        print(out.stdout[-2000:], file=sys.stderr)
        print(out.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"{binary} failed")
    times = RUNTIME_RE.findall(out.stdout + out.stderr)
    lines = [l for l in out.stdout.strip().splitlines() if l.strip() and "Run Time" not in l]
    if not times:
        print(out.stdout[-2000:], file=sys.stderr)
        raise SystemExit("no timing found — is .timer output being captured?")
    real, user, _ = times[-1]
    return float(real), float(user), (lines[-1] if lines else "")


def run_concurrent(binary, sql_path, k, cwd):
    env = dict(os.environ, DATAZOO_DISABLE_TELEMETRY="1")
    start = time.perf_counter()
    procs = [subprocess.Popen([binary, "-unsigned", "-noheader", "-list", "-f", sql_path],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, cwd=cwd, env=env) for _ in range(k)]
    outs = [p.communicate()[0] for p in procs]
    makespan = time.perf_counter() - start
    user = sum(float(RUNTIME_RE.findall(o)[-1][1]) for o in outs if RUNTIME_RE.findall(o))
    answers = {l.strip() for o in outs
               for l in o.strip().splitlines()[-1:] if l.strip()}
    return makespan, user, (answers.pop() if len(answers) == 1 else "!!DIVERGED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="label=/path/to/duckdb (repeatable; 2 arms gives a ratio)")
    ap.add_argument("--shape", choices=["wide", "tall"], default="wide")
    ap.add_argument("--features", type=int, default=500)
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--cats", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="anofox_tabfm_threads; 0 = default")
    ap.add_argument("--contend", type=int, default=0, metavar="K",
                    help="run K copies at once; report makespan and total CPU")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cwd", default=os.getcwd())
    args = ap.parse_args()

    if args.shape == "wide":
        setup, features = wide_sql(args.features, cats=args.cats), args.features
        title = f"wide: 100 rows x {args.features} features (cats={args.cats})"
    else:
        setup, features = tall_sql(args.rows, cats=args.cats), 8
        title = f"tall: {args.rows} rows x 8 features (cats={args.cats})"

    sql_path = write_sql(build_script(setup, features, args.threads))
    arms = [a.split("=", 1) for a in args.arm]
    real = {l: [] for l, _ in arms}
    cpu = {l: [] for l, _ in arms}
    answers = {}

    try:
        for r in range(args.rounds):
            for lbl, binary in arms:          # adjacent in time, alternating
                if args.contend:
                    t, u, ans = run_concurrent(binary, sql_path, args.contend, args.cwd)
                else:
                    t, u, ans = run_once(binary, sql_path, args.cwd)
                real[lbl].append(t)
                cpu[lbl].append(u)
                answers.setdefault(lbl, ans)
                print(f"  round {r+1} {lbl:10s} {t*1000:9.1f} ms   cpu {u:7.2f} s", flush=True)
    finally:
        os.unlink(sql_path)

    mode = f"{args.contend} concurrent" if args.contend else "single query"
    print(f"\n{title}   [{mode}, threads={args.threads or 'default'}, rounds={args.rounds}]")
    for lbl, _ in arms:
        print(f"  {lbl:10s} median {statistics.median(real[lbl])*1000:9.1f} ms   "
              f"cpu {statistics.median(cpu[lbl]):7.2f} s")
    if len(arms) == 2:
        a, b = arms[0][0], arms[1][0]
        print(f"  ratio {a}/{b} = {statistics.median(real[a])/statistics.median(real[b]):.3f}x"
              f"   cpu {statistics.median(cpu[a])/statistics.median(cpu[b]):.3f}x")

    distinct = set(answers.values())
    if len(distinct) == 1 and "!!DIVERGED" not in distinct:
        print(f"  answers identical across arms: {distinct.pop()}")
    else:
        print(f"  !! ANSWERS DIFFER between arms: {answers}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

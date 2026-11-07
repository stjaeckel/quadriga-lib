#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

"""Run the fuzz targets of tests/fuzzing and write a short report of what they found.

Usage (normally through "make test-fuzz", see tests/fuzzing/README.md):

    python3 tests/fuzzing/run.py --build-dir <fuzz build> --out <output folder> --time 8h [--jobs N] [--targets ...]

The corpus of each target is kept in <out>/corpus/<target> and grows from run to run. Each run writes
<out>/runs/<date_time>/ with report.md, one folder per finding, the raw fuzzer artifacts and the logs;
<out>/latest points to the newest run.
"""

import argparse
import datetime
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[2]

# Order of the findings in the report: memory errors first, resource problems last
KIND_ORDER = ["memory", "undefined-behavior", "uncaught exception", "deadly signal", "timeout", "out-of-memory", "memory-leak"]

# UBSan check types for the suppressions of known undefined behaviour; UBSan has no type that covers all checks
UB_CHECKS = ["alignment", "bool", "bounds", "enum", "float-cast-overflow", "function", "integer-divide-by-zero",
             "nonnull-attribute", "null", "object-size", "pointer-overflow", "returns-nonnull-attribute", "shift-base",
             "shift-exponent", "signed-integer-overflow", "vla-bound", "vptr"]

# Sanitizer settings of a single run of an input; the reproduce line of the report uses the same
REPRO_ENV = "OMP_NUM_THREADS=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"


def parse_time(text):
    """Seconds from "90", "90s", "10m", "8h" or "1h30m"."""
    parts = re.findall(r"(\d+)\s*([hms]?)", text.strip())
    if not parts or "".join(n + u for n, u in parts) != re.sub(r"\s", "", text):
        raise argparse.ArgumentTypeError(f"invalid time: {text}")
    return sum(int(n) * {"h": 3600, "m": 60, "s": 1, "": 1}[u] for n, u in parts)


def find_symbolizer():
    """llvm-symbolizer for readable stack traces; Debian/Ubuntu install it only with a version suffix."""
    if shutil.which("llvm-symbolizer"):
        return shutil.which("llvm-symbolizer")
    found = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        found += [p for p in Path(d).glob("llvm-symbolizer-*") if os.access(p, os.X_OK)]
    found.sort(key=lambda p: int(re.sub(r"\D", "", p.name) or 0))
    return str(found[-1]) if found else None


def fuzz_env(tmp, suppressions):
    """The environment of all target runs. The triage depends on these settings, so values from the calling shell
    (e.g. a log_path in ASAN_OPTIONS or OMP_NUM_THREADS) are replaced."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"  # One thread per fuzzing process; parallel work comes from the processes
    env["OMP_WAIT_POLICY"] = "passive"
    env["ASAN_OPTIONS"] = "detect_leaks=1"
    env["UBSAN_OPTIONS"] = f"halt_on_error=1:print_stacktrace=1:suppressions={suppressions}"
    env.pop("LSAN_OPTIONS", None)
    env["QD_FUZZ_TMP"] = str(tmp)
    sym = find_symbolizer()
    if sym:
        env.setdefault("ASAN_SYMBOLIZER_PATH", sym)
    return env


def repro_env():
    sym = find_symbolizer()
    return REPRO_ENV + (f" ASAN_SYMBOLIZER_PATH={sym}" if sym and not shutil.which("llvm-symbolizer") else "")


def limits(args):
    return [f"-timeout={args.input_timeout}", f"-rss_limit_mb={args.rss_limit_mb}", f"-malloc_limit_mb={args.rss_limit_mb}"]


# ---------------------------------------------------------------------------------------------------------------------
# Reading the sanitizer and libFuzzer output of one input

FRAME_RE = re.compile(r"^\s*#(\d+) 0x[0-9a-f]+ in (.+?) (/\S+?):(\d+)(?::\d+)?\s*$")


def short_function(name):
    """quadriga_lib::qrt_file_read from "void quadriga_lib::qrt_file_read<double>(std::string const&, ...)"."""
    out, depth = "", 0
    for ch in name:  # drop template arguments
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(depth - 1, 0)
        elif depth == 0:
            out += ch
    out = re.sub(r"\[abi:[^\]]*\]", "", out.split("(")[0]).strip()
    return out.split()[-1] if out else name


def first_stack(output):
    """Frames (function, file, line) of the first stack trace in the output."""
    frames, started = [], False
    for line in output.splitlines():
        m = FRAME_RE.match(line)
        if m:
            started = True
            frames.append((m.group(2), m.group(3), int(m.group(4))))
        elif started and line.strip() and not line.lstrip().startswith("#"):
            break
    return frames


def relative(path):
    try:
        return str(Path(path).resolve().relative_to(SOURCE_DIR))
    except ValueError:
        return None


def classify(output, returncode, wall_timeout):
    """Kind of failure and the message line that names it."""
    rules = [
        (r"ERROR: AddressSanitizer: (\S+)", lambda m: m.group(1)),
        (r"ERROR: LeakSanitizer: detected memory leaks", lambda m: "memory-leak"),
        (r"runtime error: .*", lambda m: "undefined-behavior"),
        (r"ERROR: libFuzzer: timeout", lambda m: "timeout"),
        (r"ERROR: libFuzzer: out-of-memory \(malloc\(\d+\)\)", lambda m: "out-of-memory (single allocation)"),
        (r"ERROR: libFuzzer: out-of-memory", lambda m: "out-of-memory"),
        (r"terminate called after throwing an instance of '([^']+)'", lambda m: f"uncaught exception {m.group(1)}"),
        (r"ERROR: libFuzzer: deadly signal", lambda m: "deadly signal"),
        (r"ERROR: libFuzzer: fuzz target exited", lambda m: "fuzz target exited"),
    ]
    for pattern, kind in rules:
        m = re.search(pattern, output)
        if m:
            line = next(l for l in output.splitlines() if m.group(0) in l).strip()
            return kind(m), re.sub(r"^==\d+==\s*", "", line)
    if wall_timeout:
        return "timeout", "the replay did not finish within its time limit"
    return f"exit code {returncode}", "no sanitizer or libFuzzer message"


def kind_rank(kind):
    later = [k for k in KIND_ORDER[1:] if kind.startswith(k)]
    return KIND_ORDER.index(later[0]) if later else 0


def replay(binary, input_file, args, env):
    """Run one input. None if it passes, else (signature, kind, message, frames, output)."""
    wall = 2 * args.input_timeout + 60
    try:
        p = subprocess.run([str(binary), *limits(args), str(input_file)], env=env, capture_output=True, text=True,
                           errors="replace", timeout=wall, cwd=input_file.parent)
        output, rc, timed_out = p.stdout + p.stderr, p.returncode, False
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
        rc, timed_out = None, True
    if rc == 0:
        return None
    kind, message = classify(output, rc, timed_out)
    frames = first_stack(output)
    # The frame that names the finding: library code first, then the target file, the common header last
    lib = [f for f in frames if (relative(f[1]) or "").startswith(("src/", "include/"))]
    own = [f for f in frames if relative(f[1]) and not relative(f[1]).endswith("fuzz_target.hpp")]
    other = [f for f in frames if "fuzz_target.hpp" not in f[1] and not f[1].startswith("/usr/") and Path(f[1]).exists()]
    top = (lib or own or other or frames or [None])[0]
    if top:
        signature = f"{kind} in {short_function(top[0])} ({relative(top[1]) or Path(top[1]).name})"
    else:
        signature = f"{kind} in {binary.name}"
    return signature, kind, message, frames, output


# ---------------------------------------------------------------------------------------------------------------------
# Fuzzing


def last_stats(log):
    """Runs, coverage and runs per second from the last status line of a libFuzzer fork-mode log."""
    stats = {}
    try:
        for line in Path(log).read_text(errors="replace").splitlines():
            # The exec/s field of fork mode is that of the last job only; runs over time is the average
            m = re.match(r"#(\d+):\s+cov: (\d+) ft: (\d+) corp: (\d+) exec/s:? \d+ oom/timeout/crash: (\d+)/(\d+)/(\d+) "
                         r"time: (\d+)s job: (\d+)", line)
            if m:
                runs, seconds = int(m.group(1)), max(int(m.group(8)), 1)
                stats = {"runs": runs, "cov": int(m.group(2)), "exec_s": runs // seconds, "jobs": int(m.group(9)),
                         "failed_jobs": int(m.group(5)) + int(m.group(6)) + int(m.group(7)), "elapsed": int(m.group(8))}
    except OSError:
        pass
    return stats


def schedule(targets, jobs, total_time):
    """Batches of (target, workers) and the time per batch. All targets run at once if there are enough jobs."""
    if len(targets) <= jobs:
        workers = [jobs // len(targets) + (1 if i < jobs % len(targets) else 0) for i in range(len(targets))]
        return [list(zip(targets, workers))], total_time
    batches = [targets[i:i + jobs] for i in range(0, len(targets), jobs)]
    return [[(t, 1) for t in b] for b in batches], max(total_time // len(batches), 60)


def stop_group(p, sig):
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_batch(batch, seconds, args, env, run_dir, corpus_root, info):
    procs = []
    for target, workers in batch:
        name = target.name
        corpus, artifacts = corpus_root / name, run_dir / "artifacts" / name
        corpus.mkdir(parents=True, exist_ok=True)
        artifacts.mkdir(parents=True, exist_ok=True)
        log = open(run_dir / "logs" / f"{name}.log", "w")
        seeds = subprocess.run([str(target), f"--write-seeds={corpus}"], env=env, capture_output=True, text=True, cwd=run_dir)
        log.write(seeds.stdout + seeds.stderr)
        log.flush()
        info[name] = {"corpus_before": sum(1 for _ in corpus.iterdir()), "seconds": seconds, "workers": workers,
                      "seed_error": seeds.returncode != 0}
        cmd = [str(target), f"-fork={workers}", "-ignore_crashes=1", "-ignore_timeouts=1", "-ignore_ooms=1",
               f"-max_total_time={seconds}", *limits(args), f"-artifact_prefix={artifacts}/", str(corpus)]
        # Own process group, so that the fork-mode workers end with their parent
        procs.append((name, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=run_dir,
                                             start_new_session=True), log))
    deadline = time.time() + seconds + 300
    try:
        for name, p, log in procs:
            try:
                info[name]["returncode"] = p.wait(timeout=max(deadline - time.time(), 1))
            except subprocess.TimeoutExpired:
                stop_group(p, signal.SIGKILL)
                info[name]["returncode"] = "killed after the time limit"
    except KeyboardInterrupt:
        for name, p, log in procs:
            stop_group(p, signal.SIGINT)  # libFuzzer ends its workers and exits on SIGINT
        for name, p, log in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                stop_group(p, signal.SIGKILL)
                p.wait()
        info["interrupted"] = True
    for name, p, log in procs:
        log.close()
        info[name]["corpus_after"] = sum(1 for _ in (corpus_root / name).iterdir())
        info[name].update(last_stats(run_dir / "logs" / f"{name}.log"))
        # In fork mode the exit code is 1 when inputs failed; a normal end prints "INFO: exiting"
        info[name]["finished"] = "INFO: exiting:" in (run_dir / "logs" / f"{name}.log").read_text(errors="replace")


# ---------------------------------------------------------------------------------------------------------------------
# Triage: replay, group, minimize


def select_artifacts(artifacts, limit):
    """At most limit inputs, taken from each kind (crash, leak, timeout, oom) in turn and at random within a kind.
    Random, not smallest first: the smallest inputs of an API target all decode to the same first cases."""
    by_kind = {}
    files = sorted(artifacts.glob("*"))
    random.Random(0).shuffle(files)
    for f in files:
        by_kind.setdefault(f.name.split("-")[0], []).append(f)
    chosen, queues = [], list(by_kind.values())
    while len(chosen) < limit and any(queues):
        for q in queues:
            if q and len(chosen) < limit:
                chosen.append(q.pop(0))
    return chosen, sum(len(q) for q in queues)


def minimize(fd, folder, args, env):
    """Shrink the input of a finding; keep the result only if it fails with the same signature."""
    shutil.copy(fd["raw"], folder / "input.orig")
    minimized = folder / "input.min"
    try:
        subprocess.run([str(fd["target"]), "-minimize_crash=1", f"-max_total_time={args.minimize_time}", *limits(args),
                        f"-exact_artifact_path={minimized}", str(fd["raw"])], env=env, capture_output=True,
                       timeout=args.minimize_time + 2 * args.input_timeout + 120, cwd=folder)
    except subprocess.TimeoutExpired:
        pass
    if minimized.exists() and minimized.stat().st_size < fd["raw"].stat().st_size:
        again = replay(fd["target"], minimized, args, env)
        if again and again[0] == fd["signature"]:
            shutil.copy(minimized, folder / "input")
            fd["result"] = again
    if not (folder / "input").exists():
        shutil.copy(fd["raw"], folder / "input")
        (folder / "input.orig").unlink()
    if minimized.exists():
        minimized.unlink()
    fd["input"], fd["orig_size"] = folder / "input", fd["raw"].stat().st_size
    lines = fd["result"][4].splitlines()
    (folder / "output.txt").write_text("\n".join(lines[:300]) + ("\n[... output cut ...]\n" if len(lines) > 300 else "\n"))


def parallel(fn, items, jobs):
    """fn over items in threads; on Ctrl-C or SIGTERM the queued items are dropped."""
    pool = ThreadPoolExecutor(max_workers=max(jobs, 1))
    try:
        return list(pool.map(fn, items))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=False)


def triage(targets, args, env, run_dir, known, info):
    jobs = []
    for target in targets:
        artifacts = run_dir / "artifacts" / target.name
        if artifacts.exists():
            chosen, skipped = select_artifacts(artifacts, args.max_replays)
            jobs += [(target, f) for f in chosen]
            info.setdefault(target.name, {})["not_replayed"] = skipped
    results = parallel(lambda j: replay(j[0], j[1], args, env), jobs, args.jobs)

    groups, flaky = {}, {}
    for (target, f), result in zip(jobs, results):
        if result is None:
            flaky.setdefault(target.name, []).append(f.name)
        else:
            groups.setdefault(result[0], []).append((target, f, result))

    findings, known_hits = [], []
    for signature, inputs in sorted(groups.items(), key=lambda x: (kind_rank(x[1][0][2][1]), x[0])):
        counts = {}
        for target, f, _ in inputs:
            counts[target.name] = counts.get(target.name, 0) + 1
        if signature in known:
            known_hits.append((known[signature], signature, counts))
            continue
        target, f, result = min(inputs, key=lambda x: x[1].stat().st_size)
        findings.append({"signature": signature, "counts": counts, "target": target, "raw": f, "result": result})

    for i, fd in enumerate(findings, 1):
        fd["id"] = f"F{i}"
        (run_dir / "findings" / fd["id"]).mkdir(parents=True)
    parallel(lambda fd: minimize(fd, run_dir / "findings" / fd["id"], args, env), findings, args.jobs)
    return findings, known_hits, flaky


def read_known(path):
    """known.txt: "<signature> | <note>" per line; lines starting with "#" are comments."""
    known = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                signature, _, note = line.partition("|")
                known[signature.strip()] = note.strip()
    return known


def write_suppressions(path, known):
    """UBSan suppressions for the known undefined behaviour, by function, so that fuzzing goes on past it.
    Returns the suppressed functions. Memory errors cannot be suppressed."""
    functions = sorted({m.group(1) for sig in known if (m := re.match(r"undefined-behavior in (\S+) \(", sig))})
    path.write_text("".join(f"{check}:*{fn}*\n" for fn in functions for check in UB_CHECKS))
    return functions


# ---------------------------------------------------------------------------------------------------------------------
# Report


def show_input(data):
    """The input as text if it is printable, and as a command that recreates the file."""
    out = []
    try:
        text = data.decode("utf-8")
        printable = all(c.isprintable() or c in "\n\r\t" for c in text)
    except UnicodeDecodeError:
        printable = False
    if printable and data and len(data) <= 4096:
        out += ["Input as text:", "", "```text", text.rstrip("\n"), "```", ""]
    if len(data) <= 1024:
        out += ["Recreate the input file:", "", "```bash", f"python3 -c \"open('input', 'wb').write(bytes.fromhex('{data.hex()}'))\"", "```", ""]
    return out


def duration(seconds):
    return f"{seconds} s" if seconds < 120 else f"{seconds // 60} min" if seconds < 7200 else f"{seconds / 3600:.1f} h"


def git_commit():
    try:
        rev = subprocess.run(["git", "-C", str(SOURCE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(SOURCE_DIR), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return rev + (" with uncommitted changes" if dirty else "") if rev else "unknown"
    except OSError:
        return "unknown"


def write_report(path, args, targets, info, findings, known_hits, flaky, started, load_before, stage="done"):
    finished = datetime.datetime.now()
    load_after = os.getloadavg()[0]
    new_per = {t.name: 0 for t in targets}
    known_per = dict(new_per)
    for fd in findings:
        for t, n in fd["counts"].items():
            new_per[t] += n
    for _, _, counts in known_hits:
        for t, n in counts.items():
            known_per[t] += n

    L = ["# Fuzzing report", ""]
    L += [f"- Commit: {git_commit()}", f"- Started: {started:%Y-%m-%d %H:%M}, finished: {finished:%Y-%m-%d %H:%M}",
          f"- Processes: {args.jobs}; load (1 min) before {load_before:.1f}, after {load_after:.1f}",
          f"- Limits per input: {args.input_timeout} s, {args.rss_limit_mb} MB",
          f"- Result: {len(findings)} new finding(s), {len(known_hits)} known, "
          f"{sum(len(v) for v in flaky.values())} input(s) that did not fail again", ""]
    if info.get("suppressed"):
        L += [f"- Known undefined behaviour skipped while fuzzing: {', '.join(info['suppressed'])}", ""]
    if info.get("interrupted"):
        L += ["**The run was interrupted; the fuzzing time is shorter than planned.**", ""]
    if stage == "fuzzed":
        L += ["**The triage did not run or did not finish: the failing inputs are not replayed or grouped yet.**", ""]
        L += [f"- {t.name}: {sum(1 for _ in (path.parent / 'artifacts' / t.name).glob('*'))} failing input(s) in `artifacts/{t.name}/`"
              for t in targets if (path.parent / "artifacts" / t.name).is_dir()] + [""]

    L += ["## Targets", "", "| Target | Time | Processes | Runs | Runs/s | Coverage | Corpus (start / end) | New | Known | Did not fail again |",
          "| :----- | ---: | --------: | ---: | -----: | -------: | -------------------: | --: | ----: | -----------------: |"]
    for t in targets:
        i = info.get(t.name, {})
        L.append(f"| {t.name} | {duration(i.get('elapsed', i.get('seconds', 0)))} | {i.get('workers', '-')} | {i.get('runs', '-')} | {i.get('exec_s', '-')} | "
                 f"{i.get('cov', '-')} | {i.get('corpus_before', '-')} / {i.get('corpus_after', '-')} | {new_per[t.name]} | "
                 f"{known_per[t.name]} | {len(flaky.get(t.name, []))} |")
    L += ["", "Coverage counts the instrumented code locations (edges) reached; it compares runs of the same target, not targets.", ""]

    problems = [f"- {t.name}: writing the seeds failed (see logs/{t.name}.log)" for t in targets if info.get(t.name, {}).get("seed_error")]
    problems += [f"- {t.name}: the fuzzer did not finish normally (exit code {info[t.name].get('returncode')}, see logs/{t.name}.log)"
                 for t in targets if t.name in info and not info[t.name].get("finished") and not info.get("interrupted")]
    problems += [f"- {t.name}: no input was run (see logs/{t.name}.log)"
                 for t in targets if t.name in info and not info[t.name].get("runs")]
    problems += [f"- {t.name}: {info[t.name]['failed_jobs']} failing inputs in {info[t.name]['jobs']} fuzzing jobs; each failure ends a job, "
                 "so the fuzzer hardly gets past them. Fixing the defect, or listing it in known.txt if it is undefined behaviour, lets it go further"
                 for t in targets if info.get(t.name, {}).get("jobs", 0) >= 20
                 and info[t.name]["failed_jobs"] > 0.5 * info[t.name]["jobs"]]
    if problems:
        L += ["## Problems of the run", ""] + problems + [""]

    if stage == "done":
        L += ["## New findings", ""]
        if not findings:
            L += ["None.", ""]
    for fd in findings:
        signature, kind, message, frames, _ = fd["result"]
        inputs = ", ".join(f"{t} {n}" for t, n in fd["counts"].items())
        size = fd["input"].stat().st_size
        size_note = f"{size} bytes" + (f", minimized from {fd['orig_size']}" if size < fd["orig_size"] else "")
        L += [f"### {fd['id']}: {fd['signature']}", "",
              f"- Signature (for known.txt): `{fd['signature']}`",
              f"- Message: `{message}`",
              f"- Replayed inputs that hit it: {inputs}",
              f"- Input: `findings/{fd['id']}/input` ({size_note}); full output: `findings/{fd['id']}/output.txt`",
              f"- Reproduce: `{repro_env()} {fd['target']} {' '.join(limits(args))} {fd['input']}`", ""]
        shown = [f for f in frames if relative(f[1])][:8] or frames[:8]
        if shown:
            L += ["Stack (frames in the source tree):", "", "```text"]
            L += [f"{short_function(fn)}  {relative(file) or file}:{line}" for fn, file, line in shown]
            L += ["```", ""]
        L += show_input(fd["input"].read_bytes())

    if known_hits:
        L += ["## Known findings", ""]
        L += [f"- `{sig}` ({', '.join(f'{t} {n}' for t, n in counts.items())}){': ' + note if note else ''}" for note, sig, counts in known_hits]
        L += [""]
    skipped = [f"- {t.name}: {info[t.name]['not_replayed']}" for t in targets if info.get(t.name, {}).get("not_replayed")]
    if skipped:
        L += ["## Inputs not replayed", "", f"At most {args.max_replays} failing inputs per target are replayed, chosen at random with every kind of failure in turn; "
              "the others are kept in `artifacts/`. Many failing inputs usually mean one defect that is easy to hit.", ""] + skipped + [""]
    if flaky:
        L += ["## Inputs that did not fail again", "",
              "These inputs failed while fuzzing but passed when run alone (often a timeout under load). They are kept in `artifacts/`.", ""]
        L += [f"- {t}: {', '.join(names[:5])}{' ...' if len(names) > 5 else ''}" for t, names in flaky.items()]
        L += [""]
    path.write_text("\n".join(L))


# ---------------------------------------------------------------------------------------------------------------------


def say(text):
    """Progress message; a closed output (e.g. a pipe into a killed "tail") must not end the run before the report."""
    try:
        print(text, flush=True)
    except OSError:
        pass


def on_sigterm(*_):
    """SIGTERM (e.g. from "timeout") ends the run like Ctrl-C: the fuzzers stop and a report is written."""
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", required=True, type=Path, help="build folder of the fuzz targets (ENABLE_FUZZING=ON)")
    ap.add_argument("--out", required=True, type=Path, help="output folder: corpus and reports")
    ap.add_argument("--time", type=parse_time, default="10m", help="total fuzzing time, e.g. 600, 30m, 8h (default 10m)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count(), help="fuzzing processes at the same time (default: all CPUs)")
    ap.add_argument("--targets", nargs="+", help="names of the targets to run (default: all)")
    ap.add_argument("--input-timeout", type=int, default=10, help="seconds per input before it counts as a hang (default 10)")
    ap.add_argument("--rss-limit-mb", type=int, default=2048, help="memory per input before it counts as out-of-memory (default 2048)")
    ap.add_argument("--minimize-time", type=int, default=60, help="seconds to minimize each new finding (default 60)")
    ap.add_argument("--max-replays", type=int, default=500, help="failing inputs replayed per target (default 500)")
    ap.add_argument("--known", type=Path, default=Path(__file__).with_name("known.txt"), help="list of known findings")
    args = ap.parse_args()

    targets = sorted(p for p in args.build_dir.glob("fz_*") if p.is_file() and os.access(p, os.X_OK))
    if args.targets:
        missing = set(args.targets) - {t.name for t in targets}
        if missing:
            sys.exit(f"Targets not found in {args.build_dir}: {', '.join(sorted(missing))}")
        targets = [t for t in targets if t.name in args.targets]
    if not targets:
        sys.exit(f"No fuzz targets (fz_*) in {args.build_dir}; build with ENABLE_FUZZING=ON first")
    targets = [t.resolve() for t in targets]

    # libFuzzer passes the corpus and artifact folders to its workers unquoted, and stack traces are parsed by spaces
    shm = Path("/dev/shm")
    tmp_base = shm if shm.is_dir() and os.access(shm, os.W_OK) else Path(tempfile.gettempdir())
    for what, path in [("build folder", args.build_dir.resolve()), ("output folder", args.out.resolve()),
                       ("source folder", SOURCE_DIR), ("temp folder", tmp_base)]:
        if re.search(r"\s", str(path)):
            sys.exit(f"The {what} must not contain spaces: {path}")
    signal.signal(signal.SIGTERM, on_sigterm)

    started = datetime.datetime.now()
    run_dir = args.out.resolve() / "runs" / f"{started:%Y-%m-%d_%H%M%S}"
    (run_dir / "logs").mkdir(parents=True)
    latest = args.out.resolve() / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.relative_to(args.out.resolve()))

    tmp = Path(tempfile.mkdtemp(prefix="qd_fuzz_", dir=tmp_base))
    known = read_known(args.known)
    info = {"suppressed": write_suppressions(run_dir / "ubsan_suppressions.txt", known)}
    env = fuzz_env(tmp, run_dir / "ubsan_suppressions.txt")
    load_before = os.getloadavg()[0]
    batches, seconds = schedule(targets, args.jobs, args.time)
    say(f"Fuzzing {len(targets)} target(s) in {len(batches)} batch(es) of {seconds} s; output in {run_dir}")

    report = run_dir / "report.md"
    try:
        for batch in batches:
            if info.get("interrupted"):
                break
            run_batch(batch, seconds, args, env, run_dir, args.out.resolve() / "corpus", info)
        write_report(report, args, targets, info, [], [], {}, started, load_before, stage="fuzzed")
        say("Replaying and minimizing the failing inputs ...")
        findings, known_hits, flaky = triage(targets, args, env, run_dir, known, info)
    except KeyboardInterrupt:
        if not report.exists():
            write_report(report, args, targets, info, [], [], {}, started, load_before, stage="fuzzed")
        say(f"Interrupted; report without triage: {report}")
        sys.exit(130)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    write_report(report, args, targets, info, findings, known_hits, flaky, started, load_before)
    say(f"{len(findings)} new finding(s), {len(known_hits)} known. Report: {report}")


if __name__ == "__main__":
    main()

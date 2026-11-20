#!/usr/bin/env python3
"""Fill one node with vDBF tasks: run_one.sh <R> <variant> for the lines of a
task file, as many at a time as fit in the node's memory.  Several node jobs
can share one task file; they coordinate through queue/ in this directory
(it must be on a shared file system).

    python3 pool.py [--tasks tasks.txt] [--mem-gb N] [--threads 4] [--stop-launch-h 10]

COEFF (environment, default 1e-6) selects the sweep: other values use their own
queue, queue_c<COEFF>/, and scale memory.txt by MEM_SCALE (default 0.3 at 1e-5).
QUEUE, RUNNER and MEMFILE (environment) replace queue/, run_one.sh and
memory.txt for another task family (submit_hchain.sh).

Memory.  Each task reserves memory.txt's estimate for its variant (half for
R <= 1.2 A); a running task counts as max(reservation, its resident memory).
A task starts only when its reservation fits in the budget, 90% of the node's
memory.  Biggest reservations go first, smaller ones fill the gaps.  If the
tasks together still exceed 95% of the budget, the most recently started one
is killed (least work lost) and put back with a larger reservation; a task
killed for memory by the system is retried once with twice its reservation.

Queue (queue/):
    claims/<R>_<variant>/   a task is taken (mkdir is atomic), owner file inside
    done/<R>_<variant>      finished (written on exit status 0)
    failed/<R>_<variant>    gave up; skipped until submit.sh RETRY_FAILED=1
    logs/<R>_<variant>.log  run_one.sh output
Claims of node jobs that are no longer in squeue are released at start, so
resubmitting picks up tasks a job left unfinished at its time limit.

Time.  No task starts with less than --stop-launch-h hours left (vDBF cannot
resume from a checkpoint).  On SIGTERM (time limit, scancel) the running tasks
are killed and their claims released.

Peak memory per task goes to results/logs/maxrss.txt (sampled every --poll s).
Standard library only (python3 >= 3.6)."""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FC = os.path.join(HERE, "DBF.jl", "code", "n2_pes", "frozen_core")
COEFF = os.environ.get("COEFF", "")         # empty: config.sh default, 1e-6
Q = os.path.join(HERE, os.environ.get("QUEUE") or ("queue" if COEFF in ("", "1e-6") else "queue_c" + COEFF))
# another task family (the H-chain scan) brings its own runner and memory table
RUNNER = os.path.join(HERE, os.environ.get("RUNNER", "run_one.sh"))
MEMFILE = os.path.join(HERE, os.environ.get("MEMFILE", "memory.txt"))
# memory.txt is for 1e-6; 1e-5 runs carry 4-5x fewer terms (UNO + S^2 on a laptop)
MEM_SCALE = float(os.environ.get("MEM_SCALE", "0.3" if COEFF == "1e-5" else "1"))
JOB = os.environ.get("SLURM_JOB_ID", "local")
OWNER = "{}:{}:{}".format(JOB, os.uname()[1], os.getpid())
GB = 1024 ** 3


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)
    with open(os.path.join(Q, "events.log"), "a") as fh:
        fh.write("{} [{}] {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), OWNER, msg))


def node_memory_gb():
    mb = int(os.environ.get("SLURM_MEM_PER_NODE", "0") or 0)
    if mb > 0:
        return mb / 1024
    if os.path.exists("/proc/meminfo"):
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 ** 2
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / GB


def time_left_h():
    """Hours until the job's time limit (squeue %L), or infinity outside Slurm."""
    if JOB == "local":
        return float("inf")
    try:
        s = subprocess.check_output(["squeue", "-h", "-j", JOB, "-o", "%L"]).decode().strip()
    except Exception:
        return float("inf")
    days, _, hms = s.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(days or 0) * 24 + parts[0]) + parts[1] / 60 + parts[2] / 3600


def group_rss_gb():
    """Resident memory of each process group, GB."""
    out = subprocess.check_output(["ps", "-eo", "pgid=,rss="]).decode().split("\n")
    rss = {}
    for line in out:
        f = line.split()
        if len(f) == 2:
            rss[int(f[0])] = rss.get(int(f[0]), 0) + int(f[1]) / 1024 ** 2
    return rss


def live_jobs():
    if JOB == "local":
        return None
    try:
        user = os.environ.get("USER", "")
        return set(subprocess.check_output(["squeue", "-h", "-u", user, "-o", "%A"]).decode().split())
    except Exception:
        return None


def release_stale_claims():
    jobs = live_jobs()
    for key in os.listdir(os.path.join(Q, "claims")):
        d = os.path.join(Q, "claims", key)
        try:
            owner = open(os.path.join(d, "owner")).read().strip()
        except OSError:
            owner = ""
        job, _, rest = owner.partition(":")
        if job == "local":
            alive = _pid_alive(int(rest.rpartition(":")[2] or 0))
        else:
            alive = jobs is None or job in jobs
        if not alive:
            shutil.rmtree(d, ignore_errors=True)
            log("released stale claim {} ({})".format(key, owner))


def _pid_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def claim(key):
    try:
        os.mkdir(os.path.join(Q, "claims", key))
    except FileExistsError:
        return False
    with open(os.path.join(Q, "claims", key, "owner"), "w") as fh:
        fh.write(OWNER + "\n")
    return True


def unclaim(key):
    shutil.rmtree(os.path.join(Q, "claims", key), ignore_errors=True)


def mark(kind, key, text=""):
    with open(os.path.join(Q, kind, key), "w") as fh:
        fh.write(text + "\n")


def finished(key):
    return os.path.exists(os.path.join(Q, "done", key)) or os.path.exists(os.path.join(Q, "failed", key))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=os.environ.get("TASKFILE", os.path.join(HERE, "tasks.txt")))
    p.add_argument("--mem-gb", type=float, default=None, help="budget (default 90%% of the node's memory)")
    p.add_argument("--threads", type=int, default=4, help="threads for the replay step (vDBF is serial)")
    p.add_argument("--stop-launch-h", type=float, default=10.0)
    p.add_argument("--poll", type=float, default=30.0)
    a = p.parse_args()
    for d in ("claims", "done", "failed", "logs"):
        os.makedirs(os.path.join(Q, d), exist_ok=True)
    budget = a.mem_gb or 0.9 * node_memory_gb()
    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    est = {}
    for line in open(MEMFILE):
        f = line.split("#")[0].split()
        if len(f) == 2:
            est[f[0]] = float(f[1]) * MEM_SCALE
    pending = []
    for line in open(a.tasks):
        f = line.split()
        if len(f) == 2:
            R, v = f
            gb = est.get(v, 64.0 * MEM_SCALE) * (0.5 if float(R) <= 1.2 else 1.0)
            pending.append({"R": R, "variant": v, "key": "{}_{}".format(R, v), "est": min(gb, budget), "tries": 0})
    pending.sort(key=lambda t: (-t["est"], -float(t["R"])))
    hours = time_left_h()
    t_end = time.time() + hours * 3600
    release_stale_claims()
    log("pool {}: {} tasks in {}, coeff {}, budget {:.0f} GB, {} cpus, {:.1f} h left".format(
        OWNER, len(pending), os.path.basename(a.tasks), COEFF or "1e-6", budget, cpus, hours))

    running = {}

    def stop(signum, frame):
        log("signal {}: killing {} running task(s), releasing their claims".format(signum, len(running)))
        for t in running.values():
            try:
                os.killpg(t["proc"].pid, signal.SIGKILL)
            except OSError:
                pass
            unclaim(t["key"])
        sys.exit(1)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def launch(t):
        env = dict(os.environ, HEAP_HINT="{}G".format(max(1, int(0.85 * t["est"]))))
        fh = open(os.path.join(Q, "logs", t["key"] + ".log"), "a")
        t["proc"] = subprocess.Popen(["bash", RUNNER, t["R"], t["variant"], str(a.threads)],
                                     stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        t.update(fh=fh, start=time.time(), rss=0.0, peak=0.0, killed=False)
        t["tries"] += 1
        running[t["key"]] = t
        log("start  {:<16} reserve {:5.0f} GB  (try {}, {} running)".format(t["key"], t["est"], t["tries"], len(running)))

    def record(t, rc):
        line = "{} {} status={} reserved_GB={:.0f} peak_rss_GB={:.2f} wall_h={:.2f} job={} host={} coeff={}\n".format(
            t["R"], t["variant"], rc, t["est"], t["peak"], (time.time() - t["start"]) / 3600, JOB, os.uname()[1], COEFF or "1e-6")
        with open(os.path.join(FC, "results", "logs", "maxrss.txt"), "a") as fh:
            fh.write(line)

    while True:
        # reap
        for key in list(running):
            t = running[key]
            rc = t["proc"].poll()
            if rc is None:
                continue
            t["fh"].close()
            del running[key]
            record(t, rc)
            oom = rc in (137, -9) or "OutOfMemoryError" in open(os.path.join(Q, "logs", key + ".log")).read()
            if rc == 0:
                mark("done", key)
                unclaim(key)
                log("done   {:<16} peak {:5.1f} GB, {:.2f} h".format(key, t["peak"], (time.time() - t["start"]) / 3600))
            elif (t["killed"] or oom) and t["tries"] < 3:
                t["est"] = min(budget, max(2 * t["est"], 1.5 * t["peak"]))
                pending.insert(0, t)
                log("retry  {:<16} {} (exit {}); new reservation {:.0f} GB".format(
                    key, "killed by pool" if t["killed"] else "out of memory", rc, t["est"]))
            else:
                mark("failed", key, "exit {} after {} tries; see {}/logs/{}.log".format(rc, t["tries"], os.path.basename(Q), key))
                unclaim(key)
                log("FAILED {:<16} exit {} (see {}/logs/{}.log)".format(key, rc, os.path.basename(Q), key))
        # memory
        rss = group_rss_gb()
        for t in running.values():
            t["rss"] = rss.get(t["proc"].pid, 0.0)
            t["peak"] = max(t["peak"], t["rss"])
        alive = [t for t in running.values() if not t["killed"]]
        if alive and sum(t["rss"] for t in running.values()) > 0.95 * budget:
            t = max(alive, key=lambda x: x["start"])
            t["killed"] = True
            log("memory {:.0f} GB > 95% of budget: killing the newest task {}".format(
                sum(x["rss"] for x in running.values()), t["key"]))
            os.killpg(t["proc"].pid, signal.SIGKILL)
        # launch
        can_launch = (t_end - time.time()) / 3600 > a.stop_launch_h
        if can_launch:
            used = sum(max(t["est"], t["rss"]) for t in running.values())
            for t in list(pending):
                if len(running) >= cpus:
                    break
                if finished(t["key"]):
                    pending.remove(t)
                    continue
                if used + t["est"] > budget:
                    continue
                if t["tries"] == 0 and not claim(t["key"]):
                    pending.remove(t)        # another node job has it
                    continue
                pending.remove(t)
                launch(t)
                used += t["est"]
        if not running and (not pending or not can_launch):
            break
        time.sleep(a.poll)
    left = [t["key"] for t in pending if t["tries"] > 0]
    for k in left:
        unclaim(k)
    log("pool {} finished: {}".format(OWNER, "all tasks taken" if not pending else
                                     "{} task(s) left (time limit), claims released".format(len(pending))))


if __name__ == "__main__":
    main()

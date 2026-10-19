from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from .data import atomic_json, file_hash, load_dataset
from .suite import source_hash


def contained(root, relative):
    value = Path(relative)
    if value.is_absolute() or not value.parts or any(part in ("..", "") for part in value.parts):
        raise ValueError("Expected a relative path within the queue")
    resolved = (Path(root) / value).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes the queue")
    return resolved


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("profile") not in ("smoke", "experiment"):
        raise ValueError("Invalid queue version or profile")
    if manifest.get("code_sha256") != source_hash():
        raise ValueError("Source code differs from the frozen queue")
    jobs = manifest.get("jobs", [])
    if not jobs or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Missing jobs or duplicate job IDs")
    ids = {job["id"] for job in jobs}
    outputs, datasets = set(), {}
    data = (path.parent / manifest["data"]).resolve()
    for job in jobs:
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", job["id"]):
            raise ValueError("Unsafe job ID")
        config_path = contained(path.parent, job["config"])
        output = contained(path.parent, job["output"])
        if output in outputs:
            raise ValueError("Duplicate output directory")
        outputs.add(output)
        config = json.loads(config_path.read_text())
        if file_hash(config_path) != job["config_sha256"]:
            raise ValueError("Job configuration hash mismatch")
        if config["method"] != job["method"] or config["profile"] != manifest["profile"] or config["backbone"] != job["backbone"]:
            raise ValueError("Job configuration metadata mismatch")
        if job["domain"] not in datasets:
            datasets[job["domain"]] = load_dataset(data, job["domain"])
        if datasets[job["domain"]].manifest_sha256 != job["data_manifest_sha256"]:
            raise ValueError("Dataset manifest changed")
        if type(job["seed"]) is not int or job["seed"] < 0 or job["resource"] not in ("cpu", "gpu"):
            raise ValueError("Invalid job seed or resource")
        if type(job["max_attempts"]) is not int or not 1 <= job["max_attempts"] <= 10:
            raise ValueError("max_attempts must be between 1 and 10")
        if type(job["timeout_seconds"]) not in (int, float) or not 0 < job["timeout_seconds"] <= 604800:
            raise ValueError("Invalid timeout")
        if len(job["requires"]) != len(set(job["requires"])) or not set(job["requires"]) <= ids or job["id"] in job["requires"]:
            raise ValueError("Invalid dependencies")
    pending = {job["id"]: set(job["requires"]) for job in jobs}
    done = set()
    while pending:
        ready = {key for key, dependencies in pending.items() if dependencies <= done}
        if not ready:
            raise ValueError("Cyclic dependencies")
        done.update(ready)
        pending = {key: dependencies for key, dependencies in pending.items() if key not in ready}
    return manifest


def validate_receipt(root, job, *, expected_code_hash=None):
    output = contained(root, job["output"])
    path = output / "result.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("status") not in ("complete", "no_certified_pair"):
        raise ValueError("Result is not terminal")
    for key in ("config_sha256", "data_manifest_sha256", "seed", "method", "domain"):
        if receipt.get(key) != job[key]:
            raise ValueError("Result provenance mismatch: " + key)
    if not re.fullmatch(r"[0-9a-f]{64}", receipt.get("code_sha256", "")):
        raise ValueError("Missing result code hash")
    if expected_code_hash is not None and receipt["code_sha256"] != expected_code_hash:
        raise ValueError("Result source code changed")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing artifact hashes")
    required = {"predictions.jsonl", "metrics.json", "per-session.jsonl"} if receipt["status"] == "complete" else {"selection.json"}
    if not required <= artifacts.keys():
        raise ValueError("Missing required completion artifacts")
    for name, expected in artifacts.items():
        artifact = contained(output, name)
        if artifact == path or file_hash(artifact) != expected:
            raise ValueError("Result artifact hash mismatch: " + name)
    return receipt


@contextlib.contextmanager
def queue_lock(root):
    path = Path(root) / ".queue.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("A queue coordinator is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def command(root, manifest, job):
    return [sys.executable, "-m", "rabr.run", "--config", str(contained(root, job["config"])),
            "--data", str((Path(root) / manifest["data"]).resolve()), "--domain", job["domain"],
            "--output", str(contained(root, job["output"])), "--seed", str(job["seed"])]


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_queue(path, *, cpu_workers=1, gpu_ids=(), cpu_threads=4, poll_seconds=1, retry_seconds=30, expected_code_hash=None):
    if type(cpu_workers) is not int or cpu_workers < 0 or type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Invalid worker counts")
    if len(gpu_ids) != len(set(gpu_ids)) or any(not re.fullmatch(r"[0-9]+", str(index)) for index in gpu_ids):
        raise ValueError("GPU IDs must be distinct nonnegative indices")
    if poll_seconds <= 0 or retry_seconds < 0:
        raise ValueError("Invalid poll or retry interval")
    path = Path(path).resolve()
    root = path.parent
    manifest = read_manifest(path)
    if expected_code_hash is None:
        from .run import code_hash
        expected_code_hash = code_hash()
    if any(job["resource"] == "gpu" for job in manifest["jobs"]) and not gpu_ids:
        raise ValueError("This suite requires at least one GPU worker")
    if any(job["resource"] == "cpu" for job in manifest["jobs"]) and not cpu_workers:
        raise ValueError("This suite requires at least one CPU worker")
    slots = [("cpu", str(i)) for i in range(cpu_workers)] + [("gpu", str(i)) for i in gpu_ids]
    state_dir = root / "state"
    state_dir.mkdir(exist_ok=True)
    jobs = {job["id"]: job for job in manifest["jobs"]}
    states, running = {}, {}

    def save(key, state):
        states[key] = state
        atomic_json(state_dir / (key + ".json"), state)

    with queue_lock(root):
        for key, job in jobs.items():
            state_path = state_dir / (key + ".json")
            state = json.loads(state_path.read_text()) if state_path.exists() else {"attempts": 0}
            if state.get("status") == "running" and state.get("pid") and alive(state["pid"]):
                raise RuntimeError("A previous child remains alive: " + key)
            receipt = validate_receipt(root, job, expected_code_hash=expected_code_hash)
            status = "complete" if receipt else ("failed" if state.get("attempts", 0) >= job["max_attempts"] else "pending")
            save(key, {**state, "status": status, "pid": None, "outcome": receipt["status"] if receipt else None})
        try:
            while True:
                for slot, entry in list(running.items()):
                    key, process, log, started = entry
                    timed_out = time.monotonic() - started >= jobs[key]["timeout_seconds"]
                    if timed_out:
                        stop_process(process)
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del running[slot]
                    error = "timeout" if timed_out else None
                    retryable = code != 0 or timed_out
                    try:
                        receipt = validate_receipt(root, jobs[key], expected_code_hash=expected_code_hash) if code == 0 and not timed_out else None
                    except (ValueError, OSError, KeyError) as exc:
                        receipt, error, retryable = None, str(exc), False
                    failure_path = contained(root, jobs[key]["output"]) / "failure.json"
                    if code != 0 and failure_path.exists() and failure_path.stat().st_mtime >= states[key]["started_at"]:
                        try:
                            failure = json.loads(failure_path.read_text())
                            retryable = failure.get("retryable") is True
                            error = f"{failure.get('error_type', 'Error')}: {failure.get('message', '')}"
                        except (ValueError, TypeError, AttributeError):
                            error, retryable = "invalid failure receipt", False
                    status = "complete" if receipt else ("failed" if not retryable or states[key]["attempts"] >= jobs[key]["max_attempts"] else "pending")
                    save(key, {**states[key], "status": status, "pid": None, "exit_code": code,
                               "outcome": receipt["status"] if receipt else None,
                               "error": error or (None if receipt else "missing successful receipt"),
                               "retry_after": time.time() + retry_seconds, "finished_at": time.time()})
                for key, job in jobs.items():
                    if states[key]["status"] == "pending" and any(states[dep]["status"] in ("failed", "blocked") or states[dep].get("outcome") == "no_certified_pair" for dep in job["requires"]):
                        save(key, {**states[key], "status": "blocked", "error": "dependency failed or has no calibrated size reference"})
                for slot in slots:
                    if slot in running:
                        continue
                    eligible = [key for key, job in jobs.items() if job["resource"] == slot[0] and states[key]["status"] == "pending"
                                and states[key].get("retry_after", 0) <= time.time()
                                and all(states[dep]["status"] == "complete" for dep in job["requires"])]
                    if not eligible:
                        continue
                    key = eligible[0]
                    job = jobs[key]
                    if file_hash(contained(root, job["config"])) != job["config_sha256"]:
                        raise ValueError("Configuration changed during execution")
                    output = contained(root, job["output"])
                    output.mkdir(parents=True, exist_ok=True)
                    attempt = states[key]["attempts"] + 1
                    log_dir = root / "logs" / key
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log = (log_dir / f"attempt-{attempt}.log").open("ab")
                    env = dict(os.environ)
                    env.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads), TOKENIZERS_PARALLELISM="false")
                    if slot[0] == "gpu":
                        env["CUDA_VISIBLE_DEVICES"] = slot[1]
                    else:
                        env["CUDA_VISIBLE_DEVICES"] = ""
                    started_at = time.time()
                    try:
                        process = subprocess.Popen(command(root, manifest, job), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                   env=env, start_new_session=True)
                    except OSError as error:
                        log.close()
                        terminal = isinstance(error, (FileNotFoundError, PermissionError)) or attempt >= job["max_attempts"]
                        save(key, {"status": "failed" if terminal else "pending", "attempts": attempt,
                                   "pid": None, "error": str(error), "retry_after": time.time() + retry_seconds})
                        continue
                    except BaseException:
                        log.close()
                        raise
                    save(key, {"status": "running", "attempts": attempt, "pid": process.pid,
                               "resource": slot[0], "worker": slot[1], "started_at": started_at})
                    running[slot] = (key, process, log, time.monotonic())
                if not running and all(state["status"] in ("complete", "failed", "blocked") for state in states.values()):
                    break
                time.sleep(poll_seconds)
        finally:
            for slot, (key, process, log, _) in running.items():
                stop_process(process)
                log.close()
                save(key, {**states[key], "status": "pending", "pid": None, "error": "coordinator interrupted"})
        return {"jobs": len(jobs), "complete": sum(x["status"] == "complete" for x in states.values()),
                "failed": sum(x["status"] == "failed" for x in states.values()), "blocked": sum(x["status"] == "blocked" for x in states.values())}

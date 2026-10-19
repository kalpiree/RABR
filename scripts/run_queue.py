import argparse
import json
from rabr.queue import run_queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--retry-seconds", type=float, default=30)
    args = parser.parse_args()
    result = run_queue(args.queue, cpu_workers=args.cpu_workers,
                       gpu_ids=tuple(args.gpus.split(",")) if args.gpus else (),
                       cpu_threads=args.cpu_threads, retry_seconds=args.retry_seconds)
    print(json.dumps(result))
    return int(bool(result["failed"] or result["blocked"]))


if __name__ == "__main__":
    raise SystemExit(main())

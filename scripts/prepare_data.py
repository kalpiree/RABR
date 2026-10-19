import argparse
import json
from rabr.data import prepare

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--mode", choices=("raw", "import"), default="raw")
parser.add_argument("--domains", nargs="+", default=["electronic", "clothing", "food"])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validation-fraction", type=float, default=.1)
parser.add_argument("--history-limit", type=int, default=20)
parser.add_argument("--include-pretrain-history", action="store_true")
parser.add_argument("--require-history", action="store_true")
if __name__ == "__main__":
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.domains, args.mode, args.seed, args.validation_fraction, args.history_limit,
                     include_pretrain_history=args.include_pretrain_history, require_history=args.require_history)
    print(json.dumps({domain: value["splits"] for domain, value in result.items()}, indent=2))

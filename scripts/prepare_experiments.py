import argparse
import json
from pathlib import Path
from rabr.experiments import prepare_experiments

parser = argparse.ArgumentParser()
parser.add_argument("--config", nargs="+", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--backbone-config", action="append", default=[], metavar="NAME=PATH")
if __name__ == "__main__":
    args = parser.parse_args()
    overrides = {}
    for value in args.backbone_config:
        name, path = value.split("=", 1)
        parsed = json.loads(Path(path).read_text())
        overrides[name] = parsed.get("backend", parsed)
    result = prepare_experiments(args.config, args.data, args.output, backbone_configs=overrides)
    print(json.dumps({"jobs": len(result["jobs"]), "profile": result["profile"]}))

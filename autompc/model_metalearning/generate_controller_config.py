import json
from pathlib import Path
from argparse import ArgumentParser

BASE_CONFIG_PATH = "controller_cfgs/base_config.json"
CONTROLLER_CONFIG_PATH = "controller_cfgs/"
MODEL_CONFIG_PATH = "models_json/"

def main(args):
    with open(BASE_CONFIG_PATH, "r") as f:
        base_config = json.load(f)

    model_config_path = Path(MODEL_CONFIG_PATH) / f"{args.benchmark}_port_{args.portfolio_size}_seed_{args.seed}_model.json"
    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    combined_config = {**base_config, **model_config}

    output_path = Path(CONTROLLER_CONFIG_PATH) / f"{args.benchmark}_port_{args.portfolio_size}_seed_{args.seed}.json"
    with open(output_path, "w") as f:
        json.dump(combined_config, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark", "-b")
    parser.add_argument("--portfolio_size", "-p")
    parser.add_argument("--seed", "-s")
    args = parser.parse_args()
    main(args)
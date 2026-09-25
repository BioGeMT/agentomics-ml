import argparse
import importlib.util
import sys
from pathlib import Path


def run_classifier_inference():
    script = Path(__file__).with_name("classifier_inference.py")
    spec = importlib.util.spec_from_file_location("classifier_inference", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load classifier inference from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.output.name == "eval_predictions_validation.csv":
        print("Intentional scripted validation failure")
        return 23
    run_classifier_inference()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import argparse
from argparse import ArgumentParser
from pathlib import Path


from src.experiments.run_experiments import run_experiments
config_base = Path(__file__).resolve().parent / "configs" / "models"

def _available_models_by_library() -> dict[str, list[str]]:
    libraries = ["sklearn", "torch"]
    result: dict[str, list[str]] = {}
    for lib in libraries:
        lib_dir = config_base / lib
        if lib_dir.exists():
            result[lib] = sorted(p.stem for p in lib_dir.glob("*.yaml"))
        else:
            result[lib] = []
    return result


AVAILABLE_MODELS = _available_models_by_library()
AVAILABLE_MODELS_HELP = "\n".join(
    f"  {lib}: {', '.join(models) if models else 'none'}"
    for lib, models in AVAILABLE_MODELS.items()
)

parser = ArgumentParser(
    description="Run an experiment with a registered model configuration.",
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=False,
    help=f"Model name. Available per library:\n{AVAILABLE_MODELS_HELP}",
)
parser.add_argument(
    "-lib",
    "--library",
    type=str,
    choices=["sklearn", "torch"],
    required=True,
    help="ML library backend.",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    required=False,
    help="Dataset identifier (optional).",
)


args = parser.parse_args()

if __name__ == "__main__":
    available_for_library = AVAILABLE_MODELS.get(args.library, [])
    if args.model and args.model not in available_for_library:
        parser.error(
            f"Unknown model '{args.model}' for library '{args.library}'. "
            f"Available: {', '.join(available_for_library)}"
        )
  
    config_path = config_base / args.library / f"{args.model}.yaml" 
    if args.library:
        config_path = config_base / args.library
        if args.model:
            config_path = config_path / f"{args.model}.yaml"


    if not config_path.exists():
        parser.error(f"Config file '{config_path}' not found.")

    print("---------- Running experiments ----------")

    run_experiments(config_path, args.library)
    print("\n---------- Experiments finished ----------\n")
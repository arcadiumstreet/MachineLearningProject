from .search import search
import yaml
from pathlib import Path

from src.data.data_loader import load_dataset
from src.training.evaluator import FinalEvaluator
from configs.config import RESULTS_DIR
from configs.config import CHECKPOINTS_DIR
from datetime import datetime
from src.utils.checkpoint import save_checkpoint
from src.utils.utils import save_yaml_config

def run_experiments(config_path, library):
    config_path = Path(config_path)  # Convert string to Path object
    
    # Check if config_path is a file or folder
    if config_path.is_file():
        # If it's a file, run once
        metrics = run(config_path, library)
        save_metrics(metrics)

    elif config_path.is_dir():
        # If it's a folder, iterate over each YAML file and execute run
        for config_file in config_path.glob("*.yaml"):
            metrics = run(config_file, library)
            save_metrics(metrics)
    else:
        raise ValueError(f"Config path '{config_path}' is neither a file nor a directory.")
    
    
def run(config_path, library):
    config_path = Path(config_path)  # Convert string to Path object
    cfg = yaml.safe_load(open(config_path))
    datasets = load_dataset(cfg["dataset"], test_size=0.2, transform=True)

    # LOGGING
    print(f"\n\nRunning model: {cfg['model']}\n")

    if "search" in cfg and cfg["search"]["type"] == "random":
        best_model, best_params = search(
            cfg,
            datasets,
            library
        )
        print("BEST:", best_model)
        print("BEST PARAMS:", best_params)
    (_, _), (X_test, y_test) = datasets
    
    evaluator = FinalEvaluator(X_test, library)

    y_pred = evaluator.predict(best_model, X_test)
    metrics = evaluator.get_all_metrics(y_test, y_pred)

    # # automatically save checkpoint after training
    current_datatime = datetime.now().strftime('%m-%d_%H-%M-%S')
    checkpoint_path = CHECKPOINTS_DIR / f"{current_datatime}_{cfg['model']}"
    save_checkpoint(best_model, checkpoint_path)
    save_yaml_config(checkpoint_path, **{"model": cfg["model"], 
                                        "params": best_params})


    metrics["model"] = cfg["model"]
    metrics["datetime"] = current_datatime
    return metrics


def save_metrics(metrics):
    path = RESULTS_DIR / "metrics.csv"
    # Append values; write header only when file is new/empty
    write_header = not path.exists() or path.stat().st_size == 0
    metrics.to_csv(path, mode="a", index=False, header=write_header)



if __name__ == "__main__":
    import sys
    run_experiments(sys.argv[1], sys.argv[2])

import os
import pickle
from pathlib import Path

import torch


def _resolve_model_path(path_dir: os.PathLike | str) -> Path:
    """Return the checkpoint file path, supporting .pt and .pkl."""
    base = Path(path_dir)
    pt_path = base / "model.pt"
    pkl_path = base / "model.pkl"

    if pt_path.exists():
        return pt_path
    if pkl_path.exists():
        return pkl_path
    raise FileNotFoundError(f"No checkpoint found in {base} (expected model.pt or model.pkl)")


def save_checkpoint(model, path_dir):
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "state_dict"):  # PyTorch model
        torch.save(model.state_dict(), path_dir / "model.pt")
    else:  # sklearn model or others
        with open(path_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f)


def load_checkpoint(model_class, path_dir, config=None):
    model_path = _resolve_model_path(path_dir)

    if model_path.suffix == ".pt":
        model = model_class(config)
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        with open(model_path, "rb") as f:
            return pickle.load(f)

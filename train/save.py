
import torch
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def get_next_experiment_dir(base_path: str, prefix: str) -> Path:
    base = Path(base_path)
    existing = [d for d in base.iterdir() if d.is_dir()
                and d.name.startswith(f"{prefix}_")]
    indices = [int(d.name.split('_')[-1])
               for d in existing if d.name.split('_')[-1].isdigit()]
    next_idx = max(indices, default=-1) + 1
    return base / f"{prefix}_{next_idx:04d}"


def save_experiment(
    model: torch.nn.Module,
    history: Dict[str, List[float]],
    base_path: str,
    prefix: str = "brain_tumor_mri_CNN",
    save_full_model: bool = True,
    export_torchscript: bool = False,
    remark: Optional[str] = None
) -> None:
    exp_dir = get_next_experiment_dir(base_path, prefix)
    exp_dir.mkdir(parents=True, exist_ok=False)
    print(f"Saving to: {exp_dir}")

    torch.save(model.state_dict(), exp_dir / "weights.pth")

    if save_full_model:
        torch.save(model, exp_dir / "model_full.pth")

    if export_torchscript:
        scripted = torch.jit.script(model)
        scripted.save(exp_dir / "model_scripted.pt")

    pd.DataFrame(history).to_csv(exp_dir / "history.csv", index=False)

    if remark is not None:
        with open(exp_dir / "remark.txt", "w") as f:
            f.write(remark)

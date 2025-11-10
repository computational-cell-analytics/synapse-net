import os
from glob import glob
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from elf.io import open_file
from synapse_net.training.supervised_training import get_3d_model
from synapse_net.inference.actin import segment_actin
import torch_em
import torch

def predict_actin(input_dir, model_path, output_dir, device: int=0, torch_load: bool=False, state_key: Optional[str]=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    model_name = model_path.stem 

    if torch_load:
        ckpt = str(model_path / "best.pt")
        x = torch.load(ckpt, map_location=f"cuda:{device}", weights_only=False)
        model = get_3d_model(out_channels=2)
        if state_key is None:
            state_key = "model_state"
        model.load_state_dict(x[state_key])
    else:
        model = torch_em.util.load_model(str(model_path), device=f"cuda:{device}")

    for data_path in input_dir.glob("*.h5"):
        with h5py.File(data_path, "r") as f:
            raw = f["raw"][:]
            labels = f["labels/actin"][:]

        seg, pred = segment_actin(raw, model=model, verbose=True, return_predictions=True)

        output_path = output_dir / f"{data_path.stem}.h5"

        print(f"Writing prediction to {output_path}.")
        with h5py.File(output_path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=raw, compression="gzip")
            if "labels/actin" not in f:
                f.create_dataset("labels/actin", data=labels, compression="gzip")
            f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")
            f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")

def main():
    MODEL_DIR = Path("/mnt/data1/sage/synapse-net/scripts/cryo/actin/output")
    PRED_DIR = Path("/mnt/data1/sage/synapse-net/scripts/cryo/actin/predictions")

    predict_actin(
        input_dir = "/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/test",
        model_path = MODEL_DIR / "experiment2/run3/checkpoints/actin-adapted-opto2deepict-v2",
        output_dir = PRED_DIR / "deepict",
        device = 3,
        torch_load=True,
        state_key="teacher_state"
    ) 

    predict_actin(
        input_dir = "/mnt/data1/sage/actin-segmentation/data/deepict/deepict_actin/test",
        model_path = MODEL_DIR / "experiment1/run1/checkpoints/actin-deepict-v3",
        output_dir = PRED_DIR / "deepict",
        device = 3,
        torch_load=True,
        state_key="model_state"
    )

    predict_actin(
        input_dir = "/mnt/data1/sage/actin-segmentation/data/EMPIAR-12292/h5/test",
        model_path = MODEL_DIR / "experiment1/run3/checkpoints/actin-adapted-deepict2opto-v2",
        output_dir = PRED_DIR / "opto",
        device = 3,
        torch_load=True,
        state_key="teacher_state"
    ) 

if __name__ == "__main__":
    main()

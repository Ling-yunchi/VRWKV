import os
import shutil
from datetime import datetime

import torch


def create_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    # list all run_{i} folders
    run_folders = [f for f in os.listdir(base_dir) if f.startswith("run_")]
    # get the last run_{i} folder
    if run_folders:
        last_run = sorted(run_folders)[-1]
        run_id = int(last_run.split("_")[-1]) + 1
    else:
        run_id = 0
    run_dir = os.path.join(base_dir, f"run_{run_id}")
    os.makedirs(run_dir)
    return run_dir


def save_script(run_dir, file_path):
    # add time to end of the python script name yyyy_mm_dd_hh_mm_ss
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = os.path.basename(file_path).split(".")[0] + f"_{now}.py"
    shutil.copy(file_path, os.path.join(run_dir, file_name))


def save_checkpoint(checkpoint_path, model, optimizer, loss, mean_IoU, iter_count):
    torch.save(
        {
            "iter": iter_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "loss": loss,
            "mean_IoU": mean_IoU,
        },
        checkpoint_path,
    )


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = checkpoint["iter"]
    last_loss = checkpoint["loss"]
    last_mean_IoU = checkpoint["mean_IoU"]

    print(
        f"Loaded checkpoint from iteration {start_iter} with mean IoU: {last_mean_IoU:.4f}"
    )
    return start_iter, last_loss, last_mean_IoU


def load_checkpoint_lax(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint["model_state_dict"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model_state = model.state_dict()
    matched_weights = {
        k: v
        for k, v in model_state_dict.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(matched_weights)
    model.load_state_dict(model_state)

    missing_keys = set(model_state.keys()) - set(matched_weights.keys())
    unexpected_keys = set(model_state_dict.keys()) - set(matched_weights.keys())

    if missing_keys:
        print(f"Warning: Missing keys in the loaded weights:{', '.join(missing_keys)}")
    if unexpected_keys:
        print(
            f"Warning: Unexpected keys found in the loaded weights:{', '.join(unexpected_keys)}"
        )

    return model


def load_backbone(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint["model_state_dict"]
    backbone_state_dict = {
        k: v for k, v in model_state_dict.items() if k.startswith("backbone")
    }
    model_params = model.state_dict()
    model_params.update(backbone_state_dict)
    model.load_state_dict(model_params)
    print("Loaded backbone weights from checkpoint.")

import os
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


def save_checkpoint(checkpoint_path, model, optimizer, loss, mean_IoU, iter_count):
    torch.save(
        {
            "iter": iter_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
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

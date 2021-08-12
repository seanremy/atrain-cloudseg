"""TO DO

Some code borrowed from: https://github.com/erikwijmans/skynet-ddp-slurm-example
"""
import argparse
import datetime
import json
import os
import sys
import threading
from collections.abc import Mapping
from contextlib import nullcontext

import ifcfg
import numpy as np
import torch
import torch.distributed as distrib
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.atrain import (
    ALL_METRICS,
    MASK_ONLY_METRICS,
    SQUASH_BINS_METRICS,
    ATrain,
    collate_atrain,
    get_transforms,
    interp_atrain_output,
)
from losses.factory import loss_factory
from losses.segmentation import SmoothnessPenalty
from models.single_pixel import SinglePixel
from models.unet import UNet
from utils.plotting import get_cloud_mask_viz

EXIT = threading.Event()
EXIT.clear()


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="unet", help="Which model architecture to use.")
    parser.add_argument("--base-depth", type=int, default=64, help="Base depth for the U-Net model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use for training and validation.")
    parser.add_argument("--eval-frequency", type=int, default=1, help="How often (in epochs) to evaluate.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument("--mask-only", action="store_true", help="Specify to only predict masks, and not cloud type.")
    parser.add_argument("--loss", type=str, default="cross-entropy", help="Which loss function to use.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs to run the experiment for.")
    parser.add_argument("--save-frequency", type=int, default=5, help="How often (in epochs) to checkpoint the model.")
    parser.add_argument("--squash-bins", action="store_true", help="Squash the height bins into a binary mask.")
    args = parser.parse_args()
    assert args.base_depth > 0
    assert args.batch_size > 1
    assert args.eval_frequency > 0
    assert args.learning_rate > 0
    assert args.loss in loss_factory
    assert args.num_epochs > 0
    if args.squash_bins:
        args.mask_only = True
    return args


def init_distrib_slurm(backend: str = "nccl") -> tuple[int, distrib.TCPStore]:
    """Initialize the distributed backend in a way that plays nicely with SLURM.

    Args:
        backend: The name of the backend to use, defaults to 'nccl'.

    Returns:
        local_rank: The value of the LOCAL_RANK environment variable.
        tcp_store: The TCP Store of the master process.
    """
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = ifcfg.default_interface()["device"]

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = ifcfg.default_interface()["device"]

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    tcp_store = distrib.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    distrib.init_process_group(backend, store=tcp_store, rank=world_rank, world_size=world_size)

    return local_rank, tcp_store


def convert_groupnorm_model(module: nn.Module, ngroups: int = 32) -> nn.Module:
    """Convert all of the batch-normalization layers in a module to gropu-normalization.

    Args:
        module: The module to convert.
        ngroups: The number of groups to use, defaults to 32.
    Returns:
        mod: The converted module.
    """
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))
    return mod


def dict_to(d: dict, device: torch.device) -> dict:
    """Recursively sends a dictionary's leaf nodes to the provided device, if they're torch Tensors.

    Args:
        d: The dictionary to convert.
        device: The destination device.

    Returns:
        d: The converted dictionary.
    """
    for k in d:
        if isinstance(d[k], Mapping):
            d[k] = dict_to(d[k], device)
        elif isinstance(d[k], torch.Tensor):
            d[k] = d[k].to(device)
    return d


def run_epoch(
    mode: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dloader: DataLoader,
    writer: SummaryWriter,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
) -> tuple[dict, torch.Tensor]:
    """Run the model for an epoch.

    Args:
        mode: Specifies 'train' or 'val'.
        model: The model to use for this epoch.
        optimizer: The optimizer for this model.
        dloader: The data loader.
        writer: Summary writer for tensorboard updates.
        scheduler: The learning rate scheduler.
        epoch: Which epoch this is.
        args: The command-line arguments.

    Returns:
        predictions: The dictionary of predictions.
        stats: The accumulated statistics from this epoch.
    """
    assert mode in ["train", "val"]

    if mode == "train":
        model.train()
    else:
        model.eval()

    device = next(model.parameters()).device
    context = nullcontext() if mode == "train" else torch.no_grad()
    transforms = get_transforms(mode, dloader.dataset.multi_angle_idx)

    # TO DO: remove. this just keeps the normalization transform
    transforms = [transforms[0]]

    objective = loss_factory[args.loss](125 if not args.squash_bins else 1, 9 if not args.mask_only else 2)
    penalty = SmoothnessPenalty()

    stats = torch.zeros((3,), device=device)
    pbar = tqdm(dloader)
    predictions = {}
    batch_idx = -1
    viz_freq = 50
    for batch in pbar:
        batch_idx += 1
        batch = dict_to(batch, device)
        for t in transforms:
            batch = t(batch)

        if args.squash_bins:
            batch["output"]["cloud_scenario"] = batch["output"]["cloud_scenario"].any(dim=1, keepdim=True)
        elif args.mask_only:
            batch["output"]["cloud_scenario"] = batch["output"]["cloud_scenario"] > 0

        x = batch["input"]["sensor_input"]
        y = batch["output"]["cloud_scenario"]

        with context:
            out = model(x)
            out_interp = interp_atrain_output(batch, out)
            if args.mask_only:
                pred_classes = out_interp > 0.5  # since our objective is based on (1 - mask, mask)
                out_interp = torch.stack([1 - out_interp, out_interp], dim=2)
            else:
                pred_classes = torch.argmax(out_interp.view(-1, 9), dim=-1)
            loss = objective(out_interp, batch)  # average loss per pixel, per height bin
            # loss += 0.4 * penalty(out, batch)  # TO DO: make this more customizable
            loss_total = loss * out_interp.numel()  # multiply by num elements to get total loss

        if mode == "train":
            optimizer.zero_grad()
            # loss_total.backward()
            loss.backward()
            optimizer.step()

        correct = pred_classes == y  # count of correct predictions
        cls_accs = []  # store per-class accuracy
        for cls_idx in range(out_interp.shape[-1]):
            cls_mask = y == cls_idx
            if not cls_mask.any():
                cls_accs.append(1)
            else:
                cls_accs.append((cls_mask * correct).sum() / cls_mask.sum())
        cls_accs = torch.stack(cls_accs)
        mean_acc = cls_accs.mean()

        stats[0] += loss_total
        stats[1] += mean_acc * y.size(0)
        stats[2] += y.size(0)  # number of pixels in this batch

        for i in range(batch["instance_id"].shape[0]):
            predictions[int(batch["instance_id"][i])] = pred_classes[batch["input"]["interp"]["batch_idx"] == i]

        desc_list = [
            f"Avg {mode.capitalize()} Loss={stats[0] / stats[2]:.3f}",
            f"Mean {mode.capitalize()} Acc={stats[1] / stats[2]:.3f}",
        ]
        pbar.set_description("  |  ".join(desc_list))

        writer.add_scalar(f"{mode} loss", loss_total, epoch * len(dloader) + batch_idx + 1)
        writer.add_scalar(f"{mode} acc", mean_acc, epoch * len(dloader) + batch_idx + 1)
        if (batch_idx + 1) % viz_freq == 0:
            batch_viz = []
            for i in range(batch["instance_id"].shape[0]):
                iid = batch["instance_id"][i]
                gt_cloud_scenario = batch["output"]["cloud_scenario"][batch["input"]["interp"]["batch_idx"] == i]
                viz = get_cloud_mask_viz(predictions[int(iid)], gt_cloud_scenario)
                batch_viz.append(viz)
            batch_viz = np.concatenate(batch_viz, axis=1)
            viz_tag = f"cloud mask viz | epoch {epoch + 1} | {mode} batch {batch_idx + 1}"
            writer.add_image(viz_tag, batch_viz, dataformats="HW")

        if EXIT.is_set():
            return

    distrib.all_reduce(stats)

    if distrib.get_rank() == 0:
        stats[0:2] /= stats[2]

    tqdm.write(f"{mode.capitalize()}: Avg Loss={stats[0].item():.3f}    Acc={stats[1].item():.3f}")
    if mode == "val":
        scheduler.step(stats[1])

    return predictions, stats


def make_exp_dir(args: argparse.Namespace) -> None:
    """Initialize the experiment directory with the files and subdirectories we need.

    Args:
        args: The command-line arguments.
    """
    exp_dir = os.path.join("experiments", args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        json.dump(vars(args), open(os.path.join(exp_dir, "args.json"), "w"))
    os.makedirs(os.path.join(exp_dir, "tensorboard"), exist_ok=True)
    os.makedirs(os.path.join("data", "checkpoints", args.exp_name), exist_ok=True)


def checkpoint(
    model: nn.Module,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stats: dict,
    val_metrics: dict,
) -> None:
    """Save the model checkpoint.

    Args:
        model: The model to save.
        args: The command-line arguments.
        optimizer: The optimizer for this model.
        epoch: Which epoch this is.
        stats: loss stats gathered from this epoch.
        val_metrics: the validation metrics from this epoch.
    """
    exp_dir = os.path.join("experiments", args.exp_name)
    if hasattr(model, "module"):
        model = model.module
    if (epoch + 1) % args.save_frequency == 0:
        savefile = open(os.path.join("data", "checkpoints", args.exp_name, f"epoch_{epoch+1}.pt"), "wb")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
            savefile,
        )
    os.makedirs(os.path.join(exp_dir, "stats"), exist_ok=True)
    stats = {k: v.cpu().detach().numpy().tolist() for k, v in stats.items()}
    json.dump(stats, open(os.path.join(exp_dir, "stats", f"epoch_{epoch+1}.json"), "w"))
    if not val_metrics is None:
        os.makedirs(os.path.join(exp_dir, "val_metrics"), exist_ok=True)
        val_metrics = {k: v.tolist() for k, v in val_metrics.items()}
        json.dump(val_metrics, open(os.path.join(exp_dir, "val_metrics", f"epoch_{epoch+1}.json"), "w"))


def main():
    args = parse_args()

    # local_rank, _ = init_distrib_slurm(backend="gloo")
    local_rank, _ = init_distrib_slurm()

    world_rank = distrib.get_rank()
    world_size = distrib.get_world_size()

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Have all workers wait
    distrib.barrier()

    make_exp_dir(args)

    # TO DO: don't omit any fields
    angles_to_omit = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    fields_to_omit = [
        ["Data_Directional_Fields", "I443NP"],
        ["Data_Directional_Fields", "I490P"],
        ["Data_Directional_Fields", "Q490P"],
        ["Data_Directional_Fields", "U490P"],
        ["Data_Directional_Fields", "I565NP"],
        ["Data_Directional_Fields", "I670P"],
        ["Data_Directional_Fields", "Q670P"],
        ["Data_Directional_Fields", "U670P"],
        ["Data_Directional_Fields", "I763NP"],
        ["Data_Directional_Fields", "I765NP"],
        # ["Data_Directional_Fields", "I865P"],
        ["Data_Directional_Fields", "Q865P"],
        ["Data_Directional_Fields", "U865P"],
        ["Data_Directional_Fields", "I910NP"],
        ["Data_Directional_Fields", "I1020NP"],
    ]

    # TO DO: change back to default split
    atrain_train = ATrain("train", angles_to_omit=angles_to_omit, fields_to_omit=fields_to_omit)
    atrain_val = ATrain("val", angles_to_omit=angles_to_omit, fields_to_omit=fields_to_omit)
    train_loader = DataLoader(
        atrain_train,
        batch_size=args.batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_train, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        atrain_val,
        batch_size=args.batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_val, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        num_workers=4,
    )
    metric_list = MASK_ONLY_METRICS if args.mask_only else ALL_METRICS

    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join("data", "tensorboard", f"{args.exp_name}_{now_str}")
    os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    in_channels = len(atrain_train.multi_angle_idx)
    out_channels = 125 * 9
    metric_list = ALL_METRICS
    if args.squash_bins:
        out_channels = 1
        metric_list = SQUASH_BINS_METRICS
    elif args.mask_only:
        out_channels = 125
        metric_list = MASK_ONLY_METRICS

    if args.arch == "unet":
        model = UNet(in_channels, out_channels, args.base_depth, atrain_train[0]["input"]["sensor_input"].shape[1:])
    elif args.arch == "single_pixel":
        model = SinglePixel(in_channels, out_channels)
    else:
        raise NotImplementedError(f"No implementation for model: {args.arch}")

    # # Let's use group norm instead of batch norm because batch norm can be problematic if the batch size per GPU gets really small
    # model = convert_groupnorm_model(model, ngroups=min(32, args.batch_size))
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", threshold=1e-3, eps=1e-12)

    for epoch in range(args.num_epochs):
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        tqdm.write(f"\n\n===== Epoch {epoch+1:3d}/{args.num_epochs:3d} =====")
        tqdm.write(f"learning rate: {lr:.2e}\n")

        _, train_stats = run_epoch("train", model, optimizer, train_loader, writer, scheduler, epoch, args)
        val_predictions, val_stats = run_epoch("val", model, optimizer, val_loader, writer, scheduler, epoch, args)
        val_metrics = None
        if distrib.get_rank() == 0 and (epoch + 1) % args.eval_frequency == 0:
            val_metrics = atrain_val.evaluate(val_predictions, metric_list)
            tqdm.write("\nMetrics:")
            for metric, instance_values in val_metrics.items():
                if metric == "instance_ids":
                    continue
                overall_metric = np.mean(instance_values[~np.isnan(instance_values)])
                tqdm.write(f"{metric[:min(len(metric), 30)]}:\t{overall_metric:.3f}")
                if not np.isnan(overall_metric):
                    writer.add_scalar(metric, overall_metric, (epoch + 1))

        train_loader.sampler.set_epoch(epoch)

        checkpoint(model, args, optimizer, epoch, {"train": train_stats, "val": val_stats}, val_metrics)

        if EXIT.is_set():
            break


if __name__ == "__main__":
    main()

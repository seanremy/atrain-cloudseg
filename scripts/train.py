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

import numpy as np
import torch
import torch.distributed as distrib
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
from utils.parasol_fields import FIELD_DICT
from utils.plotting import get_cloud_mask_viz
from utils.slurm import init_distrib_slurm

EXIT = threading.Event()
EXIT.clear()


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles-to-omit", type=str, default="", help="Comma-separated list of angles to omit.")
    parser.add_argument("--arch", type=str, default="unet", help="Which model architecture to use.")
    parser.add_argument("--base-depth", type=int, default=64, help="Base depth for the U-Net model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to use for training and validation.")
    parser.add_argument("--dont-weight-loss", action="store_true", help="Don't class-weight the loss function.")
    parser.add_argument("--eval-frequency", type=int, default=1, help="How often (in epochs) to evaluate.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--fields", type=str, default="directional", help="Which fields to use as input.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Starting learning rate.")
    parser.add_argument("--loss", type=str, default="cross-entropy", help="Which loss function to use.")
    parser.add_argument("--mask-only", action="store_true", help="Specify to only predict masks, and not cloud type.")
    parser.add_argument("--no-data-aug", action="store_true", help="Skip data augmentation, for debugging purposes.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs to run the experiment for.")
    parser.add_argument("--num-workers", type=int, default=20, help="Number of workers.")
    parser.add_argument("--profile", action="store_true", help="Whether or not to use the pytorch profiler.")
    parser.add_argument("--save-frequency", type=int, default=5, help="How often (in epochs) to checkpoint the model.")
    parser.add_argument("--smoothness", type=float, default=0, help="Weight for the smoothness penalty.")
    parser.add_argument("--squash-bins", action="store_true", help="Squash the height bins into a binary mask.")
    args = parser.parse_args()
    args.angles_to_omit = args.angles_to_omit.split(",")
    args.angles_to_omit = [int(a) for a in args.angles_to_omit if a != ""]
    for angle in args.angles_to_omit:
        assert angle >= 0 and angle < 16
    assert args.arch in ["unet", "single_pixel"]
    assert args.base_depth > 0
    assert args.batch_size > 1
    assert args.eval_frequency > 0
    assert args.fields in FIELD_DICT
    assert args.learning_rate > 0
    assert args.loss in loss_factory
    assert args.num_epochs > 0
    if args.squash_bins:
        args.mask_only = True
    return args


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
    profiler: torch.profiler.profile,
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
    if args.no_data_aug:
        transforms = transforms[:1]  # if no aug, keep only the normalization

    cls_weight = None
    if not args.dont_weight_loss:
        # effective sample number class weighting: https://arxiv.org/abs/1901.05555
        tc = dloader.dataset.split["train_counts"]
        cc = tc["cls_counts"]
        if args.squash_bins:
            percent_cloudy = tc["mask_count"] / tc["total_pixels"]
            cc = [1 - percent_cloudy, percent_cloudy]
        elif args.mask_only:
            cc = [cc[0], sum(cc[1:])]
        total = sum(cc)
        beta = (total - 1) / total
        inv_E_n = [(1 - beta) / (1 - beta ** count) for count in cc]
        cls_weight = [x / sum(inv_E_n) for x in inv_E_n]
        cls_weight = torch.Tensor(cls_weight).cuda()

    objective = loss_factory[args.loss](
        125 if not args.squash_bins else 1, 9 if not args.mask_only else 2, weight=cls_weight
    )
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
            if args.smoothness > 0:
                loss += args.smoothness * penalty(out, batch)
            loss_total = loss * out_interp.numel()  # multiply by num elements to get total loss

        if mode == "train":
            optimizer.zero_grad()
            # loss_total.backward()
            loss.backward()
            optimizer.step()
        if args.profile:
            profiler.step()

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
        ] + [f"size: {batch['input']['sensor_input'].size()}"]
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
    fields = FIELD_DICT[args.fields]
    atrain_train = ATrain("train", angles_to_omit=args.angles_to_omit, fields=fields)
    atrain_val = ATrain("val", angles_to_omit=args.angles_to_omit, fields=fields)
    train_loader = DataLoader(
        atrain_train,
        batch_size=args.batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_train, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        atrain_val,
        batch_size=args.batch_size,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_val, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        num_workers=args.num_workers,
    )
    metric_list = MASK_ONLY_METRICS if args.mask_only else ALL_METRICS

    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join("data", "tensorboard", f"{args.exp_name}_{now_str}")
    os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    profiler = None
    if args.profile:
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_dir),
            record_shapes=True,
            with_stack=True,
        )

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

        _, train_stats = run_epoch("train", model, optimizer, train_loader, writer, scheduler, profiler, epoch, args)
        val_predictions, val_stats = run_epoch(
            "val", model, optimizer, val_loader, writer, scheduler, profiler, epoch, args
        )
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

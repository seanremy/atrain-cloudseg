"""TO DO

Some code borrowed from: https://github.com/erikwijmans/skynet-ddp-slurm-example
"""
import argparse
import json
import os
import sys
import threading
from collections.abc import Mapping
from contextlib import nullcontext

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.atrain import ALL_METRICS, MASK_ONLY_METRICS, ATrain, collate_atrain, interp_atrain_output
from models.unet import UNet

EXIT = threading.Event()
EXIT.clear()


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_depth", type=int, default=64, help="Base depth for the U-Net model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use for training and validation.")
    parser.add_argument("--eval-frequency", type=int, default=1, help="How often (in epochs) to evaluate.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Starting learning rate.")
    parser.add_argument("--mask-only", action="store_true", help="Specify to only predict masks, and not cloud type.")
    parser.add_argument("--num-epochs", type=int, default=200, help="Number of epochs to run the experiment for.")
    args = parser.parse_args()
    assert args.base_depth > 0
    assert args.batch_size > 1
    assert args.eval_frequency > 0
    assert args.learning_rate > 0
    assert args.num_epochs > 0
    return args


def get_ifname():
    """TO DO"""
    return ifcfg.default_interface()["device"]


def init_distrib_slurm(backend="nccl"):
    """TO DO"""
    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    master_port = int(os.environ.get("MASTER_PORT", 8738))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    world_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))

    tcp_store = distrib.TCPStore(master_addr, master_port, world_size, world_rank == 0)
    distrib.init_process_group(backend, store=tcp_store, rank=world_rank, world_size=world_size)

    return local_rank, tcp_store


def convert_groupnorm_model(module, ngroups=32):
    """TO DO"""
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))
    return mod


def dict_to(d, device):
    """TO DO"""
    for k in d:
        if isinstance(d[k], Mapping):
            d[k] = dict_to(d[k], device)
        elif isinstance(d[k], torch.Tensor):
            d[k] = d[k].to(device)
    return d


def run_epoch(mode, model, optimizer, dloader, scheduler, epoch, mask_only):
    """TO DO"""
    assert mode in ["train", "val"]

    device = next(model.parameters()).device
    context = nullcontext() if mode == "train" else torch.no_grad()

    stats = torch.zeros((3,), device=device)
    pbar = tqdm(dloader)
    predictions = {}
    for batch in pbar:
        batch = dict_to(batch, device)

        x = batch["input"]["sensor_input"]
        y = batch["output"]["cloud_scenario"].view(-1)

        with context:
            out = model(x)
            out_interp = interp_atrain_output(batch, out)
            if mask_only:
                out_sig = F.sigmoid(out_interp)
                loss = F.binary_cross_entropy(out_sig.view(-1), (y > 0).float(), reduction="sum")
                pred_classes = out_sig > 0.5
            else:
                out_flat = out_interp.view(-1, 9)
                loss = F.cross_entropy(out_flat, y, reduction="sum")
                pred_classes = torch.argmax(out_flat, dim=-1)

        if mode == "train":
            optimizer.zero_grad()
            (loss / y.size(0)).backward()
            optimizer.step()

        stats[0] += loss
        stats[1] += (pred_classes.view(-1) == y).float().sum()
        stats[2] += y.size(0)
        avg_loss = stats[0] / stats[2]

        for i in range(batch["instance_id"].shape[0]):
            predictions[int(batch["instance_id"][i])] = pred_classes[batch["input"]["interp"]["batch_idx"] == i]

        pbar.set_description(
            f"{mode.capitalize()} Loss={loss / y.size(0):.3f}  |  Avg {mode.capitalize()} Loss={avg_loss:.3f}"
        )

        if EXIT.is_set():
            return

    distrib.all_reduce(stats)

    if distrib.get_rank() == 0:
        stats[0:2] /= stats[2]
        pbar.set_description(f"{mode.capitalize()}: Avg Loss={stats[0].item():.3f}    Acc={stats[1].item():.3f}")
        if mode == "val":
            scheduler.step(stats[2])

    return predictions, stats


def checkpoint(args, model, optimizer, epoch, stats, val_metrics):
    """TO DO"""
    exp_dir = os.path.join("experiments", args.exp_name)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        json.dump(vars(args), open(os.path.join(exp_dir, "args.json"), "w"))
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    model_path = os.path.join(exp_dir, "models", f"epoch_{epoch+1}.pt")
    if hasattr(model, "module"):
        model = model.module
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
        model_path,
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

    atrain_train = ATrain("train")
    atrain_val = ATrain("val")
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

    out_channels = 125 if args.mask_only else 125 * 9
    model = UNet(240, out_channels, args.base_depth, atrain_train[0]["input"]["sensor_input"].shape[1:])
    # Let's use group norm instead of batch norm because batch norm can be problematic if the batch size per GPU gets really small
    model = convert_groupnorm_model(model, ngroups=min(32, args.batch_size))
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    for epoch in range(args.num_epochs):
        tqdm.write(f"\n===== Epoch {epoch+1:3d} =====\n")
        _, train_stats = run_epoch("train", model, optimizer, train_loader, scheduler, epoch, args.mask_only)
        val_predictions, val_stats = run_epoch("val", model, optimizer, val_loader, scheduler, epoch, args.mask_only)
        val_metrics = None
        if distrib.get_rank() == 0 and (epoch + 1) % args.eval_frequency == 0:
            val_metrics = atrain_val.evaluate(val_predictions, metric_list)
            tqdm.write("\nMetrics:")
            for m in val_metrics:
                tqdm.write(f"{m[:min(len(m), 30)]}:\t{val_metrics[m]:.3f}")

        train_loader.sampler.set_epoch(epoch)

        checkpoint(args, model, epoch, {"train": train_stats, "val": val_stats}, val_metrics)

        if EXIT.is_set():
            break


if __name__ == "__main__":
    main()

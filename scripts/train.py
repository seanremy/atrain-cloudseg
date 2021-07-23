"""TO DO

Some code borrowed from: https://github.com/erikwijmans/skynet-ddp-slurm-example
"""

import os
import sys
import threading
from collections.abc import Mapping

import ifcfg
import torch
import torch.distributed as distrib
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if "./src" not in sys.path:
    sys.path.insert(0, "./src")  # TO DO: change this once it's a package
from datasets.atrain import ATrain, collate_atrain, interp_atrain_output
from models.unet import UNet

EXIT = threading.Event()
EXIT.clear()


def get_ifname():
    return ifcfg.default_interface()["device"]


def init_distrib_slurm(backend="nccl"):
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
    mod = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        mod = nn.GroupNorm(ngroups, module.num_features, affine=module.affine)
    for name, child in module.named_children():
        mod.add_module(name, convert_groupnorm_model(child, ngroups))

    return mod


def dict_to(d, device):
    for k in d:
        if isinstance(d[k], Mapping):
            d[k] = dict_to(d[k], device)
        elif isinstance(d[k], torch.Tensor):
            d[k] = d[k].to(device)
    return d


def train_epoch(model, optimizer, dloader, epoch):
    """TO DO"""
    device = next(model.parameters()).device

    tqdm.write(f"\n===== Epoch {epoch+1:3d} =====\n")
    train_stats = torch.zeros((3,), device=device)
    num_batches = len(dloader)
    pbar = tqdm(dloader)
    for batch_idx, batch in enumerate(pbar):
        batch = dict_to(batch, device)

        x = batch["input"]["sensor_input"]
        y = batch["output"]["cloud_scenario"].view(-1)

        out = model(x)
        out_interp = interp_atrain_output(batch, out)
        out_flat = out_interp.view(-1, 9)

        loss = F.cross_entropy(out_flat, y, reduction="sum")

        optimizer.zero_grad()
        (loss / x.size(0)).backward()
        optimizer.step()

        train_stats[0] += loss
        train_stats[1] += (torch.argmax(out_flat, -1) == y).float().sum()
        train_stats[2] += x.size(0)
        avg_loss = train_stats[0] / train_stats[2]

        pbar.set_description(f"Train Loss: {loss:.3f}  |  Avg Train Loss: {avg_loss:.3f}")

        if EXIT.is_set():
            return

    distrib.all_reduce(train_stats)

    if distrib.get_rank() == 0:
        train_stats[0:2] /= train_stats[2]
        tqdm.write(f"Train: Loss={train_stats[0].item():.3f}    Acc={train_stats[1].item():.3f}")


def eval_epoch(model, dloader):
    """TO DO"""
    device = next(model.parameters()).device

    eval_stats = torch.zeros((3,), device=device)
    num_batches = len(dloader)
    pbar = tqdm(dloader)
    for batch_idx, batch in enumerate(pbar):
        batch = dict_to(batch, device)

        x = batch["input"]["sensor_input"]
        y = batch["output"]["cloud_scenario"].view(-1)

        with torch.no_grad():
            out = model(x)
            out_interp = interp_atrain_output(batch, out).to(device)
            out_flat = out_interp.view(-1, 9)
            loss = F.cross_entropy(out_flat, y, reduction="sum")

        pbar.set_description(f"Val Loss: {loss:.3f}  |  Avg Val Loss: {avg_loss:.3f}")

        eval_stats[0] += loss
        eval_stats[1] += (torch.argmax(out_flat, -1) == y).float().sum()
        eval_stats[2] += x.size(0)

        if EXIT.is_set():
            return

    distrib.all_reduce(eval_stats)

    if distrib.get_rank() == 0:
        eval_stats[0:2] /= eval_stats[2]
        tqdm.write(f"Val:   Loss={eval_stats[0].item():.3f}    Acc={eval_stats[1].item():.3f}")


def main():
    BASE_DEPTH = 64
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 200

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
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_train, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        atrain_val,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset=atrain_val, num_replicas=world_size, rank=world_rank
        ),
        collate_fn=collate_atrain,
        num_workers=4,
    )

    model = UNet(240, 125 * 9, BASE_DEPTH, atrain_train[0]["input"]["sensor_input"].shape[1:])
    # Let's use group norm instead of batch norm because batch norm can be problematic if the batch size per GPU gets really small
    model = convert_groupnorm_model(model, ngroups=min(32, BATCH_SIZE))
    model = model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device],
        output_device=device,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_epoch(model, optimizer, train_loader, epoch)
        eval_epoch(model, val_loader)

        train_loader.sampler.set_epoch(epoch)

        if EXIT.is_set():
            break


if __name__ == "__main__":
    main()

import os

import ifcfg
import torch.distributed as distrib


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

"""Tensor transfer benchmark.

This script measures the transfer rate of Pytorch tensors copied between
the host (CPU) and the device (GPU). The printed results are the medians
of the measurements.
"""

import time
from statistics import median

import torch


def warmup(num_bytes: int, iters: int):
    """Copies data around the host and the device to warm-up hardware."""
    src_tensor = torch.ones(num_bytes, dtype=torch.int8)
    imed_tensor = torch.zeros(num_bytes, dtype=torch.int8, pin_memory=True)
    dst_tensor = torch.zeros(num_bytes, dtype=torch.int8, device="cuda")

    for _ in range(iters):
        imed_tensor.copy_(src_tensor)
        dst_tensor.copy_(imed_tensor)
        src_tensor.copy_(dst_tensor)
    return


def print_result(descr: str, sec: float, gbs: float):
    description = f"{descr:<17}"
    duration = f"{sec*1e3:>10.2f} ms" if sec > 10e-3 else f"{sec*1e6:>10.2f} us"
    transfer_rate = f"{gbs:>8.1f} GB/s" if gbs < 1e3 else ""

    print(f"    {description} {duration} {transfer_rate}")
    return


def tensor_to_device_dtype(
    src_tensor: torch.Tensor, device: str, dtype, descr: str, iters: int
):
    """Uses `tensor.to(device, dtype)` to create a copy of src."""
    durations = []
    for _ in range(iters):
        time_start = time.time()
        dst_tensor = src_tensor.to(device, dtype, copy=True)
        durations.append(time.time() - time_start)

    duration = median(durations)
    nbytes = dst_tensor.numel() * dst_tensor.element_size()
    transfer_rate = nbytes / 1024**3 / duration
    print_result(descr, duration, transfer_rate)


def tensor_copy(
    src_tensor: torch.Tensor, dst_tensor: torch.Tensor, descr: str, iters: int
):
    """Uses `tensor.copy_(src)` to copy elements from src into dst.

    dst may be of different data type or reside on a different device."""
    durations = []
    for _ in range(iters):
        time_start = time.time()
        dst_tensor.copy_(src_tensor)
        durations.append(time.time() - time_start)

    duration = median(durations)
    nbytes = dst_tensor.numel() * dst_tensor.element_size()
    transfer_rate = nbytes / 1024**3 / duration
    print_result(descr, duration, transfer_rate)


def profile_tensor(num_bytes: int, iters: int):
    # tensor.to() does not allow dst to be set as pinned.
    # Therefore tests such as "... to hp" do not exist.

    # Prepare all containers in advance
    t_host = torch.ones(num_bytes, dtype=torch.int8)
    t_host_pinned = torch.ones(num_bytes, dtype=torch.int8, pin_memory=True)
    t_device = torch.ones(num_bytes, dtype=torch.int8, device="cuda")

    print("\ntensor.to(device, dtype)")

    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(t_host, dst_device, dst_dtype, "h to d", iters)

    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(t_host_pinned, dst_device, dst_dtype, "hp to d", iters)

    dst_device = "cpu"
    dst_dtype = torch.int8
    tensor_to_device_dtype(t_device, dst_device, dst_dtype, "d to h", iters)

    print("\ntensor.copy_(src)")

    tensor_copy(t_host, t_device, "h to d", iters)
    tensor_copy(t_host_pinned, t_device, "hp to d", iters)
    tensor_copy(t_device, t_host, "d to h", iters)
    tensor_copy(t_device, t_host_pinned, "d to hp", iters)


def main(num_bytes: int, iters: int):
    print()
    print(f"Data size: {num_bytes / 1024 / 1024} MB")
    print(f"Iterations: {iters}")
    print("Abbreviations: d-device, h-host, p-pinned")

    profile_tensor(num_bytes, iters)


if __name__ == "__main__":
    warmup(num_bytes=64 * 1024**2, iters=200)
    main(num_bytes=16 * 1024**2, iters=10001)
    main(num_bytes=1 * 1024**2, iters=10001)

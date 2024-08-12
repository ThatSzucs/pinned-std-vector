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
    description = f"{descr:<16}"
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


def profile_tensor_to_device_dtype(num_bytes: int, iters: int):
    # tensor.to() does not care whether the other is pinned or pageable.
    # Therefore tests such as "... to hp" do not exist.

    print("\ntensor.to(device, dtype)")

    src_t = torch.ones(num_bytes, dtype=torch.int8)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "h to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, pin_memory=True)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "hp to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, device="cuda")
    dst_device = "cpu"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "d to h", iters)


def profile_tensor_copy(num_bytes: int, iters: int):
    print("\ntensor.copy_(src)")

    src_t = torch.ones(num_bytes, dtype=torch.int8)
    dst_t = torch.zeros(num_bytes, dtype=torch.int8, device="cuda")
    tensor_copy(src_t, dst_t, "h to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, pin_memory=True)
    dst_t = torch.zeros(num_bytes, dtype=torch.int8, device="cuda")
    tensor_copy(src_t, dst_t, "hp to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, device="cuda")
    dst_t = torch.zeros(num_bytes, dtype=torch.int8)
    tensor_copy(src_t, dst_t, "d to h", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, device="cuda")
    dst_t = torch.zeros(num_bytes, dtype=torch.int8, pin_memory=True)
    tensor_copy(src_t, dst_t, "d to hp", iters)


def main(num_bytes: int, iters: int):
    print()
    print(f"Data size: {num_bytes / 1024 / 1024} MB")
    print(f"Iterations: {iters}")
    print("Abbreviations: d-device, h-host, p-pinned")

    profile_tensor_to_device_dtype(num_bytes, iters)
    profile_tensor_copy(num_bytes, iters)


if __name__ == "__main__":
    warmup(num_bytes=64 * 1024**2, iters=200)
    main(num_bytes=16 * 1024**2, iters=10001)
    main(num_bytes=1 * 1024**2, iters=10001)

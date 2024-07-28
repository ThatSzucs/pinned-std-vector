"""Tensor transfer benchmark testing binded `std::vector`s too.

This script measures the transfer rate of Pytorch tensors copied between
the host (CPU) and the device (GPU). The printed results are the medians
of the measurements.
"""

import time
from statistics import median

import torch

import pyplayground as pg


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


def vector_to_ndarray(
    src_vector: pg.PageableI8Vector | pg.PinnedI8Vector, descr: str, iters: int
):
    """Calls vector.as_ndarray()."""
    durations = []
    for _ in range(iters):
        time_start = time.time()
        src_array = src_vector.as_ndarray()
        durations.append(time.time() - time_start)

    duration = median(durations)
    transfer_rate = 0
    print_result(descr, duration, transfer_rate)


def vector_to_tensor(
    src_vector: pg.PageableI8Vector | pg.PinnedI8Vector, descr: str, iters: int
):
    """Calls vector.as_ndarray() and torch.from_numpy() consecutively."""
    durations = []
    for _ in range(iters):
        time_start = time.time()
        src_array = src_vector.as_ndarray()
        src_tensor = torch.from_numpy(src_array)
        durations.append(time.time() - time_start)

    duration = median(durations)
    transfer_rate = 0
    print_result(descr, duration, transfer_rate)


def profile_tensor(num_bytes: int, iters: int):
    print("\n--- tensor ---")

    print("\ntensor.to(device, dtype)")

    src_t = torch.ones(num_bytes, dtype=torch.int8)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "h to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, pin_memory=True)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "hp to d", iters)

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


def profile_vector(num_bytes: int, iters: int):
    print("\n--- vector ---")    

    print("\ntensor.to(device, dtype)")

    src_vec = pg.PageableI8Vector(num_bytes)
    src_array = src_vec.as_ndarray()
    src_t = torch.from_numpy(src_array)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "v to d", iters)

    src_vec = pg.PinnedI8Vector(num_bytes)
    src_array = src_vec.as_ndarray()
    src_t = torch.from_numpy(src_array)
    dst_device = "cuda"
    dst_dtype = torch.int8
    tensor_to_device_dtype(src_t, dst_device, dst_dtype, "vp to d", iters)

    print("\ntensor.copy_(src)")

    src_vec = pg.PageableI8Vector(num_bytes)
    src_array = src_vec.as_ndarray()
    src_t = torch.from_numpy(src_array)
    dst_t = torch.zeros(num_bytes, dtype=torch.int8, device="cuda")
    tensor_copy(src_t, dst_t, "v to d", iters)

    src_vec = pg.PinnedI8Vector(num_bytes)
    src_array = src_vec.as_ndarray()
    src_t = torch.from_numpy(src_array)
    dst_t = torch.zeros(num_bytes, dtype=torch.int8, device="cuda")
    tensor_copy(src_t, dst_t, "vp to d", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, device="cuda")
    dst_vec = pg.PageableI8Vector(num_bytes)
    dst_array = dst_vec.as_ndarray()
    dst_t = torch.from_numpy(dst_array)
    tensor_copy(src_t, dst_t, "d to v", iters)

    src_t = torch.ones(num_bytes, dtype=torch.int8, device="cuda")
    dst_vec = pg.PinnedI8Vector(num_bytes)
    dst_array = dst_vec.as_ndarray()
    dst_t = torch.from_numpy(dst_array)
    tensor_copy(src_t, dst_t, "d to vp", iters)

    print("\ncast")

    src_vec = pg.PageableI8Vector(num_bytes)
    vector_to_ndarray(src_vec, "v as ndarray", iters)
    src_vec = pg.PageableI8Vector(num_bytes)
    vector_to_tensor(src_vec, "v as tensor", iters)

    src_vec = pg.PinnedI8Vector(num_bytes)
    vector_to_ndarray(src_vec, "vp as ndarray", iters)
    src_vec = pg.PinnedI8Vector(num_bytes)
    vector_to_tensor(src_vec, "vp as tensor", iters)


def main(num_bytes: int, iters: int):
    print()
    print(f"Data size: {num_bytes / 1024 / 1024} MB")
    print(f"Iterations: {iters}")
    print("Abbreviations: d-device, h-host, p-pinned, v-vector")

    profile_tensor(num_bytes, iters)
    profile_vector(num_bytes, iters)


if __name__ == "__main__":
    warmup(num_bytes=64 * 1024**2, iters=20)
    main(num_bytes=16 * 1024**2, iters=400)
    main(num_bytes=1 * 1024**2, iters=2001)

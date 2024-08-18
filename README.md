# pinned-std-vector

Supplementary code for the blogpost [_Accelerating data transfer with CUDA devices using pinned memory_](https://thatszucs.github.io/pinned-std-vector/).

The C++ library requires CMake and the CUDA Toolkit. The Python benchmarks require Pytorch and the bindings of the C++ library. Install the latter with:

```
pip install .
```

The project was developed using:
- Kubuntu 22.04
- Nvidia driver 550.67
- CUDA Toolkit 12.4.131
- gcc 12.3.0
- g++ 12.3.0
- Python 3.10.9
- PyTorch 2.3.1 with CUDA 12.1
- Pybind 2.5.0
- CMake 3.21.3
- Nvidia RTX 4080 (PCI Express x16 Gen4)
- Ryzen 5900x
- 32 GB 3200 MHz DDR4 RAM

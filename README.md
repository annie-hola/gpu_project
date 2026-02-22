# MNIST Digit Classification with CUDA MLP

GPU-accelerated MLP for MNIST with CPU/GPU training and performance comparison.

## Contributors

- **Kieu Anh HA**
- **Daniel BEQAJ**

## Project Objectives

- Build custom CUDA kernels for forward/backward propagation
- Compare CPU vs. GPU training performance
- Apply GPU optimization techniques (tiling, shared memory)
- Practice CUDA and parallel computing concepts

## Project Structure

```
gpu_project/
├── src/
│   ├── main.cu              # Main entry point with training loop
│   ├── mlp.c                # CPU MLP implementation
│   ├── mlp_cuda.cu          # GPU MLP implementation
│   ├── cuda_kernels.cu      # CUDA kernel implementations
│   ├── mnist_loader.c       # MNIST dataset loader
│   └── utils.cu             # Utilities (timing, memory, etc.)
├── include/
│   ├── mlp.h                # CPU MLP interface
│   ├── mlp_cuda.h           # GPU MLP interface
│   ├── cuda_kernels.h       # CUDA kernel declarations
│   ├── mnist_loader.h       # Dataset loader interface
│   └── utils.h              # Utility functions
├── data/                    # MNIST dataset files (downloaded)
├── build/                   # Compiled object files
├── bin/                     # Executable binary
├── REPORT.pdf               # Detailed report (PDF)
├── Makefile                 # Build system
└── README.md               # This file
```

## Quick Start

```bash
make download-data
make
make run
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make` | Build the project |
| `make run` | CPU + GPU comparison |
| `make run-cpu` | CPU-only training |
| `make run-gpu` | GPU-only training |
| `make benchmark` | Batch-size sweep |
| `make clean` | Remove build artifacts |
| `make distclean` | Remove all generated files |
| `make download-data` | Download MNIST dataset |
| `make check-cuda` | Verify CUDA installation |
| `make help` | List targets |

## Notes

- Default run: 10 epochs, batch size 64, hidden size 256, learning rate 0.01.
- For full details, see REPORT.pdf.

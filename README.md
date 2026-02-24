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

## GPU Optimization Experiments

Run systematic experiments to measure GPU tuning impact:

```bash
# Run all optimization experiments (takes ~30 minutes)
chmod +x run_experiments.sh
./run_experiments.sh

# Analyze results
chmod +x analyze_experiments.sh
./analyze_experiments.sh
```

**Experiments included:**
1. **THREADS_PER_BLOCK_1D** (128, 256, 512) - 1D kernel thread count impact
2. **THREADS_PER_BLOCK_2D** (8×8, 16×16, 32×32) - Matrix operation block size impact
3. **Hidden Layer Size** (128, 256, 512, 1024) - Network capacity vs. GPU scalability
4. **Learning Rate** (0.001, 0.01, 0.1) - Convergence speed & numerical stability
5. **EVAL_BATCH_SIZE** (64, 256, 512) - Evaluation throughput optimization

Results saved to `result/experiments/`

## Notes

- Default run: 10 epochs, batch size 64, hidden size 256, learning rate 0.01.
- For full details, see REPORT.pdf.

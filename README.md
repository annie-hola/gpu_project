# MNIST Digit Classification with CUDA MLP

A high-performance Multi-Layer Perceptron (MLP) implementation for MNIST digit classification, featuring both CPU and GPU (CUDA) implementations with comprehensive performance analysis.

## Project Objectives

This project implements a neural network training system to explore GPU acceleration for deep learning:

1. **Custom CUDA Kernel Development**: Build kernels for forward/backward propagation (matrix multiplication, ReLU, softmax, SGD)
2. **Performance Analysis**: Compare CPU vs. GPU execution times and measure speedup
3. **Optimization Strategies**: Implement shared memory, tiling, and coalesced memory access patterns
4. **Educational Framework**: Understand GPU programming fundamentals and parallel computing

## Architecture

### Network Structure
- **Input Layer**: 784 neurons (28×28 MNIST images)
- **Hidden Layer**: 256 neurons with ReLU activation (configurable)
- **Output Layer**: 10 neurons with Softmax (digits 0-9)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss Function**: Cross-Entropy Loss

### Implementation Components

```
┌─────────────────────────────────────────┐
│         MNIST Dataset Loader            │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼──────┐  ┌──────▼──────┐
│  CPU MLP    │  │  GPU MLP    │
│             │  │             │
│ - Forward   │  │ - CUDA      │
│ - Backward  │  │   Kernels   │
│ - SGD       │  │ - Device    │
│             │  │   Memory    │
└─────────────┘  └─────────────┘
       │                │
       └───────┬────────┘
               │
       ┌───────▼────────┐
       │  Performance   │
       │  Comparison    │
       └────────────────┘
```

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
├── docs/                    # Additional documentation
├── Makefile                 # Build system
├── README.md               # This file
└── Docs.md                 # Team workflow and task division
```

## Getting Started

### Prerequisites

- **CUDA Toolkit**: Version 11.0 or higher
- **NVIDIA GPU**: Compute Capability 7.5+ recommended
- **GCC/G++**: Version 7.0 or higher
- **Make**: Build automation
- **wget or curl**: For downloading MNIST dataset

#### Check CUDA Installation

```bash
make check-cuda
```

This will verify:
- NVCC compiler availability
- NVIDIA driver installation
- GPU detection

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/gpu_project
   ```

2. **Download MNIST dataset**:
   ```bash
   make download-data
   ```
   
   This downloads and extracts the MNIST dataset to the `data/` directory.

3. **Build the project**:
   ```bash
   make
   ```
   
   This compiles all source files and creates the executable in `bin/mnist_mlp`.

### Running the Project

#### Run both CPU and GPU training (comparison mode):
```bash
make run
# Or directly:
./bin/mnist_mlp --mode both --epochs 10 --batch-size 64
```

#### Run CPU-only training:
```bash
make run-cpu
# Or directly:
./bin/mnist_mlp --mode cpu --epochs 10 --batch-size 64
```

#### Run GPU-only training:
```bash
make run-gpu
# Or directly:
./bin/mnist_mlp --mode gpu --epochs 10 --batch-size 64
```

## Configuration Options

The program supports various command-line arguments:

```bash
./bin/mnist_mlp [OPTIONS]

Options:
  --mode <cpu|gpu|both>     Training mode (default: both)
  --epochs <N>              Number of training epochs (default: 10)
  --batch-size <N>          Batch size (default: 64)
  --hidden-size <N>         Hidden layer size (default: 256)
  --learning-rate <F>       Learning rate (default: 0.01)
  --data-dir <PATH>         Path to MNIST data (default: ./data)
  --help                    Show help message
```

### Example Usage

```bash
# Train with larger batch size
./bin/mnist_mlp --mode gpu --batch-size 128 --epochs 20

# Train with different hidden layer size
./bin/mnist_mlp --mode both --hidden-size 512 --learning-rate 0.005

# Quick test with fewer epochs
./bin/mnist_mlp --mode gpu --epochs 3
```

## Performance Benchmarking

Run comprehensive benchmarks across different batch sizes:

```bash
make benchmark
```

This will execute training with batch sizes: 32, 64, 128, 256 and report:
- Training time per epoch
- Total training time
- Final accuracy
- CPU vs GPU speedup

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make` or `make all` | Build the project |
| `make run` | Build and run (CPU + GPU comparison) |
| `make run-cpu` | Build and run CPU-only version |
| `make run-gpu` | Build and run GPU-only version |
| `make download-data` | Download MNIST dataset |
| `make benchmark` | Run performance benchmarks |
| `make clean` | Remove build artifacts |
| `make distclean` | Remove all generated files |
| `make check-cuda` | Verify CUDA installation |
| `make help` | Display help message |

## Customization

### Adjust GPU Architecture

In the `Makefile`, modify `CUDA_ARCH` based on your GPU:

```makefile
# For RTX 30xx series (Ampere)
CUDA_ARCH = -arch=sm_86

# For RTX 20xx series (Turing)
CUDA_ARCH = -arch=sm_75

# For GTX 10xx series (Pascal)
CUDA_ARCH = -arch=sm_61
```

Check your GPU's compute capability: [NVIDIA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

## Implementation Status

### Completed (Skeleton)
- [x] Project structure and build system
- [x] Header file interfaces
- [x] MNIST data loader framework
- [x] CPU/GPU training loop structure
- [x] Timing and utility functions
- [x] Main entry point with argument parsing

### 🔨 To Be Implemented (See Docs.md)
- [ ] CPU forward/backward propagation
- [ ] CUDA matrix multiplication kernels
- [ ] CUDA activation function kernels
- [ ] CUDA gradient computation kernels
- [ ] Memory optimization (shared memory, tiling)
- [ ] Performance profiling and analysis
- [ ] Gradient verification
- [ ] Advanced optimizations

## Key Concepts

### Forward Propagation
1. Linear transformation: `hidden = input @ W1 + b1`
2. ReLU activation: `hidden = max(0, hidden)`
3. Linear transformation: `output = hidden @ W2 + b2`
4. Softmax normalization: `output = exp(output) / sum(exp(output))`

### Backward Propagation
1. Output gradient: `dL/doutput = output - one_hot(labels)`
2. Hidden layer gradient: `dL/dhidden = dL/doutput @ W2^T`
3. ReLU gradient: `dL/dhidden *= (hidden > 0)`
4. Weight gradients: `dW1 = input^T @ dL/dhidden`, `dW2 = hidden^T @ dL/doutput`

### SGD Update
```
W = W - learning_rate * dW
b = b - learning_rate * db
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size 32`
- Reduce hidden layer size: `--hidden-size 128`

### Compilation Errors
- Verify CUDA toolkit installation: `nvcc --version`
- Check GPU compute capability matches `CUDA_ARCH` in Makefile
- Ensure GCC version is compatible with CUDA version

### Dataset Not Found
- Run `make download-data` to download MNIST
- Or manually place MNIST files in the `data/` directory

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

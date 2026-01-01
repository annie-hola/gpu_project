# MNIST CUDA MLP - Quick Reference

## Quick Start Commands

### CPU-Only Build (Current - No CUDA Required)
```bash
# Download MNIST dataset (if not already downloaded)
make download-data

# Build and compile CPU version
make clean && make

# Run training on CPU (10 epochs)
make run

# Run training with custom settings (5 epochs)
make run-train

# Clean build artifacts
make clean
```

### GPU Build (When CUDA Available)
```bash
# Switch to GPU Makefile
make -f Makefile.gpu check-cuda

# Build GPU version
make -f Makefile.gpu

# Run with both CPU and GPU
make -f Makefile.gpu run

# Run GPU only
make -f Makefile.gpu run-gpu

# Run CPU only
make -f Makefile.gpu run-cpu
```

### Command Line Options
```bash
./bin/mnist_mlp --mode cpu --epochs 10 --batch-size 64 --learning-rate 0.01 --hidden-size 256
```

## File Structure Overview

```
gpu_project/
├── src/                    # Implementation files
│   ├── main.cu            # Entry point (DONE - CPU mode active, GPU commented)
│   ├── mlp.c              # CPU MLP implementation (DONE - all functions)
│   ├── mlp_cuda.cu        # GPU MLP wrapper (skeleton - for future GPU work)
│   ├── cuda_kernels.cu    # CUDA kernels (skeleton - for future GPU work)
│   ├── mnist_loader.c     # MNIST data loader (DONE - loads IDX files)
│   ├── utils_cpu.c        # CPU utilities (DONE - timers, accuracy, weights)
│   └── utils.cu           # GPU utilities (skeleton - for future GPU work)
├── include/               # Header files (ALL DONE)
├── data/                  # MNIST dataset (DOWNLOADED - 60K train, 10K test)
├── bin/                   # Compiled binaries
│   └── mnist_mlp          # CPU-trained MLP executable
├── Makefile              # CPU-only build (ACTIVE - uses GCC)
├── Makefile.gpu          # GPU build (for future use - requires CUDA/nvcc)
├── README.md             # Project documentation
├── QUICKSTART.md         # This file
└── DOCS.md               # Team workflow guide
```

## Current Project Status

### Completed (CPU Implementation)
- Full 2-layer MLP (784→256→10) on CPU
- Forward propagation with ReLU and softmax
- Backward propagation with full gradient computation
- SGD optimizer with configurable learning rate
- Cross-entropy loss computation
- MNIST data loading from IDX format
- Training loop with epoch tracking
- Test set evaluation and accuracy measurement
- Build system for CPU-only compilation
- All utility functions (timers, weight init, accuracy)

### Ready for GPU Work (When GPU Access Available)
- GPU code in main.cu is commented out but ready
- All CUDA header files and API definitions prepared
- Makefile.gpu ready for CUDA compilation
- Skeleton implementations for:
  - mlp_cuda.cu (GPU MLP wrapper)
  - cuda_kernels.cu (matrix multiply, activation kernels)
  - utils.cu (GPU utilities)

### Pending GPU Implementation
- CUDA kernel for matrix multiplication
- CUDA kernel for ReLU activation
- CUDA kernel for softmax
- CUDA kernel for loss computation
- CUDA kernel for backpropagation
- GPU memory management and transfers
- Performance benchmarking CPU vs GPU

## Key Implementation Notes

### Matrix Dimensions
- Input: [batch_size, 784]
- W1: [784, hidden_size]
- Hidden: [batch_size, hidden_size]
- W2: [hidden_size, 10]
- Output: [batch_size, 10]

### Memory Layout
- Row-major order (C-style)
- Element at (i,j) in matrix M[rows][cols]: M[i * cols + j]

### CUDA Launch Config Example
```cuda
dim3 blockDim(16, 16);
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
kernel<<<gridDim, blockDim>>>(...);
```

## Key Resources

- **README.md**: Complete project documentation and architecture
- **DOCS.md**: Detailed task breakdown for team development
- **Makefile**: CPU-only build commands (current)
- **Makefile.gpu**: GPU build commands (future use)
- **include/*.h**: Complete API interfaces and function signatures
- **src/main.cu**: Entry point with CPU code active, GPU code commented

## Next Steps

### To Continue Development:
1. **Test CPU Implementation**: Run `make run` and verify accuracy improves
2. **Access Remote GPU**: Set up Google Colab, AWS, or other cloud GPU
3. **Uncomment GPU Code**: In main.cu, uncomment GPU sections
4. **Implement CUDA Kernels**: Fill in cuda_kernels.cu with actual implementations
5. **Switch to GPU Build**: Use `make -f Makefile.gpu` when ready
6. **Benchmark Performance**: Compare CPU vs GPU training times

### Expected Results (CPU):
- Epoch 1: Loss ~2.3, Accuracy ~10-20%
- Epoch 5: Loss ~0.5, Accuracy ~85-90%
- Epoch 10: Loss ~0.3, Accuracy ~92-95%
- Final test accuracy: ~92-95% on MNIST

## Important Notes

### Current Setup (CPU-Only)
1. **No CUDA Required**: The default Makefile uses GCC and works without CUDA toolkit
2. **Training Time**: CPU training takes several minutes (10 epochs ~5-10 min)
3. **Dataset Location**: MNIST files are in `data/` directory
4. **Binary Output**: Compiled executable is `bin/mnist_mlp`

### Future GPU Setup
1. **GPU Architecture**: Update `CUDA_ARCH` in Makefile.gpu for your GPU (e.g., sm_86 for RTX 30xx)
2. **CUDA Toolkit**: Install CUDA toolkit 11.0+ before using Makefile.gpu
3. **Uncomment GPU Code**: In `src/main.cu`, uncomment GPU sections when ready
4. **Remote GPU Options**: Google Colab, AWS EC2, Lambda Labs, Vast.ai

### Build System Details
- `Makefile`: CPU-only build using GCC with `-x c` flag to compile .cu as C
- `Makefile.gpu`: GPU build using NVCC (requires CUDA installation)
- Both produce same binary: `bin/mnist_mlp`

## Troubleshooting

### CPU Build Issues
- **Clang CUDA errors**: Use `make clean && make` to rebuild with `-x c` flag
- **Data not found**: Run `make download-data` or check `data/` directory exists
- **Link errors**: Ensure `-lm` is at the end of link command (math library)

### Future GPU Build Issues
- **CUDA not found**: Install CUDA toolkit and add to PATH
- **Architecture mismatch**: Update CUDA_ARCH in Makefile.gpu
- **Out of memory**: Reduce batch size (try 32 or 16)
- **Runtime errors**: Use `CUDA_CHECK()` macro for debugging
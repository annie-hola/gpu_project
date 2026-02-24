#ifndef CUDA_CONSTANTS_H
#define CUDA_CONSTANTS_H

// ==================== KERNEL CONFIGURATION ====================
// Thread block sizes for different kernel types
#define THREADS_PER_BLOCK_1D 256        // For 1D kernels (element-wise ops, reductions)
#define THREADS_PER_BLOCK_2D 16         // For 2D kernels (matrix operations)

// ==================== MATRIX MULTIPLICATION ====================
// Tile size for tiled matrix multiplication (shared memory optimization)
#define TILE_SIZE 16                    // 16x16 tiles for shared memory optimization

// ==================== EVALUATION ====================
// Batch size for evaluation (inference)
#define EVAL_BATCH_SIZE 256             // Process 256 samples at a time during evaluation

// ==================== NUMERICAL STABILITY ====================
// Small epsilon value to prevent division by zero
#define EPSILON 1e-7f                   // Used in softmax normalization

#endif // CUDA_CONSTANTS_H

#include "cuda_kernels.h"
#include <stdio.h>
#include <math.h>

// ==================== FORWARD PROPAGATION KERNELS ====================

// Basic matrix multiplication kernel
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // TODO: Implement basic matrix multiplication
    // C[M x N] = A[M x K] @ B[K x N]
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication with shared memory
__global__ void matmul_tiled_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // TODO: Implement tiled matrix multiplication using shared memory
    // Use __shared__ memory to reduce global memory accesses
    
    #define TILE_SIZE 16
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiled_col = t * TILE_SIZE + threadIdx.x;
        int tiled_row = t * TILE_SIZE + threadIdx.y;

        if (row < M && tiled_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tiled_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiled_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Add bias to each row
__global__ void add_bias_kernel(float *input, float *bias, float *output, int rows, int cols) {
    // TODO: Add bias vector to each row of the matrix
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / cols;
    int col = idx % cols;
    
    if (row < rows && col < cols) {
        output[idx] = input[idx] + bias[col];
    }
}

// ReLU activation
__global__ void relu_kernel(float *input, float *output, int size) {
    // TODO: Implement ReLU: output = max(0, input)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Softmax activation (numerically stable version)
__global__ void softmax_kernel(float *input, float *output, int batch_size, int num_classes) {
    // TODO: Implement softmax for each sample in batch
    // For numerical stability, subtract max before exp
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    if (threadIdx.x == 0) {
        float *row = &input[batch_idx * num_classes];

        float max_val = row[0];
        for (int i = 1; i < num_classes; i++) {
            if (row[i] > max_val) max_val = row[i];
        }

        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float v = expf(row[i] - max_val);
            output[batch_idx * num_classes + i] = v;
            sum += v;
        }

        float inv_sum = 1.0f / sum;
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] *= inv_sum;
        }
    }
}

// ==================== BACKWARD PROPAGATION KERNELS ====================

// Softmax + Cross-Entropy gradient (combined for efficiency)
__global__ void softmax_cross_entropy_gradient_kernel(float *output, int *labels,
                                                      float *grad_output,
                                                      int batch_size, int num_classes) {
    // TODO: Compute gradient: grad = output - one_hot(labels)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / num_classes;
    int class_idx = idx % num_classes;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        float grad = output[idx];
        if (class_idx == labels[batch_idx]) {
            grad -= 1.0f;
        }
        grad_output[idx] = grad;
    }
}

// ReLU backward
__global__ void relu_backward_kernel(float *grad_output, float *hidden,
                                     float *grad_hidden, int size) {
    // TODO: Gradient through ReLU: grad_input = grad_output if input > 0 else 0
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_hidden[idx] = (hidden[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Matrix multiplication with transpose options
__global__ void matmul_transpose_kernel(float *A, float *B, float *C,
                                        int M, int N, int K,
                                        bool transpose_A, bool transpose_B) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = transpose_A ? A[k * M + row] : A[row * K + k];
            float b_val = transpose_B ? B[col * K + k] : B[k * N + col];
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

__global__ void sum_columns_kernel(float *input, float *output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; row++) {
            sum += input[row * cols + col];
        }
        output[col] = sum;
    }
}

// ==================== OPTIMIZATION KERNELS ====================

// SGD weight update
__global__ void sgd_update_kernel(float *weights, float *gradients,
                                  float learning_rate, int size, int batch_size) {
    // TODO: Implement SGD update: weights -= learning_rate * (gradients / batch_size)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * (gradients[idx] / batch_size);
        gradients[idx] = 0.0f;  // Reset gradient
    }
}

// ==================== UTILITY KERNELS ====================

// Parallel reduction sum
__global__ void reduce_sum_kernel(float *input, float *output, int size) {
    // TODO: Implement parallel reduction to sum array elements
    // Use shared memory for efficiency
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Compute accuracy
__global__ void compute_accuracy_kernel(float *predictions, int *labels,
                                        int *correct, int batch_size, int num_classes) {
    // TODO: Count correct predictions
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Find argmax
        int pred_class = 0;
        float max_val = predictions[batch_idx * num_classes];
        
        for (int i = 1; i < num_classes; i++) {
            float val = predictions[batch_idx * num_classes + i];
            if (val > max_val) {
                max_val = val;
                pred_class = i;
            }
        }
        
        if (pred_class == labels[batch_idx]) {
            atomicAdd(correct, 1);
        }
    }
}

// Cross-entropy loss computation
__global__ void cross_entropy_loss_kernel(float *output, int *labels,
                                          float *loss, int batch_size, int num_classes) {
    // TODO: Compute cross-entropy loss
    // loss = -sum(log(output[labels[i]]))
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int label = labels[batch_idx];
        float sample_loss = -logf(output[batch_idx * num_classes + label] + 1e-10f);
        atomicAdd(loss, sample_loss);
    }
}

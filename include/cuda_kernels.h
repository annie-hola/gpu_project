#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <cuda_runtime.h>

// Forward Propagation Kernels
__global__ void matmul_kernel(float *A, float *B, float *C, 
                              int M, int N, int K);

__global__ void matmul_tiled_kernel(float *A, float *B, float *C,
                                    int M, int N, int K);

__global__ void add_bias_kernel(float *input, float *bias, float *output,
                                int rows, int cols);

__global__ void relu_kernel(float *input, float *output, int size);

__global__ void softmax_kernel(float *input, float *output, 
                               int batch_size, int num_classes);

// Backward Propagation Kernels
__global__ void softmax_cross_entropy_gradient_kernel(float *output, int *labels,
                                                      float *grad_output,
                                                      int batch_size, int num_classes);

__global__ void relu_backward_kernel(float *grad_output, float *hidden,
                                     float *grad_hidden, int size);

__global__ void matmul_transpose_kernel(float *A, float *B, float *C,
                                        int M, int N, int K, bool transpose_A, bool transpose_B);

__global__ void sum_columns_kernel(float *input, float *output, int rows, int cols);

// Weight Update Kernels (SGD)
__global__ void sgd_update_kernel(float *weights, float *gradients,
                                  float learning_rate, int size, int batch_size);

// Utility Kernels
__global__ void reduce_sum_kernel(float *input, float *output, int size);

__global__ void compute_accuracy_kernel(float *predictions, int *labels,
                                        int *correct, int batch_size, int num_classes);

// Loss computation
__global__ void cross_entropy_loss_kernel(float *output, int *labels,
                                          float *loss, int batch_size, int num_classes);

#endif // CUDA_KERNELS_H

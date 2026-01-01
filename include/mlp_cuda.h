#ifndef MLP_CUDA_H
#define MLP_CUDA_H

#include "mlp.h"
#include <cuda_runtime.h>

// GPU MLP Structure
typedef struct {
    MLPConfig config;
    
    // Device (GPU) Memory
    float *d_W1, *d_b1;
    float *d_W2, *d_b2;
    
    // Device intermediate values
    float *d_hidden, *d_output;
    float *d_input;
    int *d_labels;
    
    // Device gradients
    float *d_dW1, *d_db1;
    float *d_dW2, *d_db2;
    float *d_dhidden;
    
    // Temporary buffers
    float *d_temp;
} MLPCuda;

// GPU Implementation Functions
MLPCuda* mlp_create_cuda(MLPConfig config);
void mlp_destroy_cuda(MLPCuda *mlp);
void mlp_copy_weights_to_device(MLPCuda *mlp_cuda, MLP *mlp_cpu);
void mlp_copy_weights_to_host(MLPCuda *mlp_cuda, MLP *mlp_cpu);

// GPU Forward/Backward
void mlp_forward_cuda(MLPCuda *mlp, float *d_input, int batch_size);
void mlp_backward_cuda(MLPCuda *mlp, float *d_input, int *d_labels, int batch_size);
void mlp_update_weights_cuda(MLPCuda *mlp, int batch_size);
float mlp_compute_loss_cuda(MLPCuda *mlp, int *d_labels, int batch_size);

// Training utilities
void mlp_train_cuda(MLPCuda *mlp, float *train_data, int *train_labels,
                    int num_samples, int batch_size, int epochs);
float mlp_evaluate_cuda(MLPCuda *mlp, float *test_data, int *test_labels, int num_samples);

#endif // MLP_CUDA_H

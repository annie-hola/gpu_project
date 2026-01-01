#include "mlp_cuda.h"
#include "cuda_kernels.h"
#include "utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

MLPCuda* mlp_create_cuda(MLPConfig config) {
    MLPCuda *mlp = (MLPCuda*)malloc(sizeof(MLPCuda));
    mlp->config = config;
    
    // Allocate device memory for weights
    CUDA_CHECK(cudaMalloc(&mlp->d_W1, config.input_size * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_b1, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_W2, config.hidden_size * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_b2, config.output_size * sizeof(float)));
    
    // Allocate device memory for intermediate values (will size based on batch)
    mlp->d_hidden = NULL;
    mlp->d_output = NULL;
    mlp->d_input = NULL;
    mlp->d_labels = NULL;
    
    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&mlp->d_dW1, config.input_size * config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_db1, config.hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_dW2, config.hidden_size * config.output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&mlp->d_db2, config.output_size * sizeof(float)));
    mlp->d_dhidden = NULL;
    mlp->d_temp = NULL;
    
    return mlp;
}

void mlp_destroy_cuda(MLPCuda *mlp) {
    cudaFree(mlp->d_W1);
    cudaFree(mlp->d_b1);
    cudaFree(mlp->d_W2);
    cudaFree(mlp->d_b2);
    if (mlp->d_hidden) cudaFree(mlp->d_hidden);
    if (mlp->d_output) cudaFree(mlp->d_output);
    if (mlp->d_input) cudaFree(mlp->d_input);
    if (mlp->d_labels) cudaFree(mlp->d_labels);
    if (mlp->d_dhidden) cudaFree(mlp->d_dhidden);
    cudaFree(mlp->d_dW1);
    cudaFree(mlp->d_db1);
    cudaFree(mlp->d_dW2);
    cudaFree(mlp->d_db2);
    if (mlp->d_temp) cudaFree(mlp->d_temp);
    free(mlp);
}

void mlp_copy_weights_to_device(MLPCuda *mlp_cuda, MLP *mlp_cpu) {
    // TODO: Copy weights from CPU to GPU
    MLPConfig cfg = mlp_cuda->config;
    CUDA_CHECK(cudaMemcpy(mlp_cuda->d_W1, mlp_cpu->W1, 
                         cfg.input_size * cfg.hidden_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mlp_cuda->d_b1, mlp_cpu->b1,
                         cfg.hidden_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mlp_cuda->d_W2, mlp_cpu->W2,
                         cfg.hidden_size * cfg.output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(mlp_cuda->d_b2, mlp_cpu->b2,
                         cfg.output_size * sizeof(float),
                         cudaMemcpyHostToDevice));
}

void mlp_copy_weights_to_host(MLPCuda *mlp_cuda, MLP *mlp_cpu) {
    // TODO: Copy weights from GPU to CPU
    MLPConfig cfg = mlp_cuda->config;
    CUDA_CHECK(cudaMemcpy(mlp_cpu->W1, mlp_cuda->d_W1,
                         cfg.input_size * cfg.hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mlp_cpu->b1, mlp_cuda->d_b1,
                         cfg.hidden_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mlp_cpu->W2, mlp_cuda->d_W2,
                         cfg.hidden_size * cfg.output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(mlp_cpu->b2, mlp_cuda->d_b2,
                         cfg.output_size * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

void mlp_forward_cuda(MLPCuda *mlp, float *d_input, int batch_size) {
    // TODO: Implement GPU forward propagation using CUDA kernels
    // 1. Matrix multiply: d_hidden = d_input @ d_W1
    // 2. Add bias: d_hidden += d_b1
    // 3. ReLU activation
    // 4. Matrix multiply: d_output = d_hidden @ d_W2
    // 5. Add bias: d_output += d_b2
    // 6. Softmax
    
    MLPConfig cfg = mlp->config;
    
    // Allocate intermediate buffers if needed
    if (!mlp->d_hidden) {
        // TODO: Allocate based on max batch size
    }
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cfg.hidden_size + blockDim.x - 1) / blockDim.x,
                 (batch_size + blockDim.y - 1) / blockDim.y);
    
    // TODO: Launch kernels
    // matmul_kernel<<<gridDim, blockDim>>>(...);
    // add_bias_kernel<<<...>>>(...);
    // relu_kernel<<<...>>>(...);
    // ... continue for second layer
}

void mlp_backward_cuda(MLPCuda *mlp, float *d_input, int *d_labels, int batch_size) {
    // TODO: Implement GPU backward propagation
    // 1. Compute output gradient (softmax + cross-entropy derivative)
    // 2. Compute dW2 and db2
    // 3. Backprop to hidden layer
    // 4. ReLU gradient
    // 5. Compute dW1 and db1
}

void mlp_update_weights_cuda(MLPCuda *mlp, int batch_size) {
    // TODO: Launch SGD update kernels
    // Update all weights and biases using gradients
    
    MLPConfig cfg = mlp->config;
    int threadsPerBlock = 256;
    
    // Update W1
    int numBlocks = (cfg.input_size * cfg.hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    // sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(mlp->d_W1, mlp->d_dW1, 
    //                                                   cfg.learning_rate, 
    //                                                   cfg.input_size * cfg.hidden_size,
    //                                                   batch_size);
    
    // TODO: Update b1, W2, b2
}

float mlp_compute_loss_cuda(MLPCuda *mlp, int *d_labels, int batch_size) {
    // TODO: Compute cross-entropy loss on GPU
    // Return loss value to CPU
    return 0.0f;
}

void mlp_train_cuda(MLPCuda *mlp, float *train_data, int *train_labels,
                    int num_samples, int batch_size, int epochs) {
    // TODO: Implement GPU training loop
    // Similar to CPU version but with device memory transfers
    
    printf("Training MLP on GPU...\n");
    
    // Allocate device memory for input batch
    if (!mlp->d_input) {
        CUDA_CHECK(cudaMalloc(&mlp->d_input, batch_size * mlp->config.input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mlp->d_labels, batch_size * sizeof(int)));
    }
    
    Timer timer;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        timer_start(&timer);
        float total_loss = 0.0f;
        int num_batches = num_samples / batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            // TODO: Copy batch to device
            // TODO: Forward, backward, update
        }
        
        double epoch_time = timer_stop(&timer);
        printf("Epoch %d/%d - Loss: %.4f - Time: %.2f ms\n",
               epoch + 1, epochs, total_loss / num_batches, epoch_time);
    }
}

float mlp_evaluate_cuda(MLPCuda *mlp, float *test_data, int *test_labels, int num_samples) {
    // TODO: Evaluate on GPU
    return 0.0f;
}

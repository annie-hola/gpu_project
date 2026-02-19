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
    MLPConfig cfg = mlp->config;
    
    // Allocate intermediate buffers if needed
    if (!mlp->d_hidden) {
        CUDA_CHECK(cudaMalloc(&mlp->d_hidden, batch_size * cfg.hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mlp->d_output, batch_size * cfg.output_size * sizeof(float)));
    }
    
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cfg.hidden_size + blockDim.x - 1) / blockDim.x,
                 (batch_size + blockDim.y - 1) / blockDim.y);
    
    // Layer 1: d_hidden = d_input @ d_W1
    matmul_kernel<<<gridDim, blockDim>>>(d_input, mlp->d_W1, mlp->d_hidden,
                                         batch_size, cfg.hidden_size, cfg.input_size);

    int hidden_size = batch_size * cfg.hidden_size;
    int threads = 256;
    int blocks = (hidden_size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(mlp->d_hidden, mlp->d_b1, mlp->d_hidden,
                                         batch_size, cfg.hidden_size);
    relu_kernel<<<blocks, threads>>>(mlp->d_hidden, mlp->d_hidden, hidden_size);

    // Layer 2: d_output = d_hidden @ d_W2
    dim3 gridDim2((cfg.output_size + blockDim.x - 1) / blockDim.x,
                  (batch_size + blockDim.y - 1) / blockDim.y);
    matmul_kernel<<<gridDim2, blockDim>>>(mlp->d_hidden, mlp->d_W2, mlp->d_output,
                                          batch_size, cfg.output_size, cfg.hidden_size);

    int output_size = batch_size * cfg.output_size;
    blocks = (output_size + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(mlp->d_output, mlp->d_b2, mlp->d_output,
                                         batch_size, cfg.output_size);

    // Softmax
    softmax_kernel<<<batch_size, 1>>>(mlp->d_output, mlp->d_output,
                                      batch_size, cfg.output_size);
}

void mlp_backward_cuda(MLPCuda *mlp, float *d_input, int *d_labels, int batch_size) {
    MLPConfig cfg = mlp->config;

    if (!mlp->d_dhidden) {
        CUDA_CHECK(cudaMalloc(&mlp->d_dhidden, batch_size * cfg.hidden_size * sizeof(float)));
    }

    if (!mlp->d_temp) {
        CUDA_CHECK(cudaMalloc(&mlp->d_temp, batch_size * cfg.output_size * sizeof(float)));
    }

    // 1. Compute output gradient (softmax + cross-entropy derivative)
    int total = batch_size * cfg.output_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softmax_cross_entropy_gradient_kernel<<<blocks, threads>>>(mlp->d_output, d_labels,
                                                               mlp->d_temp,
                                                               batch_size, cfg.output_size);

    // 2. Compute dW2 and db2
    dim3 blockDim(16, 16);
    dim3 gridDim((cfg.output_size + blockDim.x - 1) / blockDim.x,
                 (cfg.hidden_size + blockDim.y - 1) / blockDim.y);
    matmul_transpose_kernel<<<gridDim, blockDim>>>(mlp->d_hidden, mlp->d_temp,
                                                   mlp->d_dW2,
                                                   cfg.hidden_size, cfg.output_size,
                                                   batch_size, true, false);

    int threads_1d = 256;
    int blocks_1d = (cfg.output_size + threads_1d - 1) / threads_1d;
    CUDA_CHECK(cudaMemset(mlp->d_db2, 0, cfg.output_size * sizeof(float)));
    sum_columns_kernel<<<blocks_1d, threads_1d>>>(mlp->d_temp, mlp->d_db2,
                                            batch_size, cfg.output_size);

    // 3. Backprop to hidden layer: dhidden = doutput @ W2^T
    dim3 gridDim2((cfg.hidden_size + blockDim.x - 1) / blockDim.x,
                  (batch_size + blockDim.y - 1) / blockDim.y);
    matmul_transpose_kernel<<<gridDim2, blockDim>>>(mlp->d_temp, mlp->d_W2,
                                                    mlp->d_dhidden,
                                                    batch_size, cfg.hidden_size,
                                                    cfg.output_size, false, true);

    // 4. ReLU gradient
    int hidden_total = batch_size * cfg.hidden_size;
    blocks = (hidden_total + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(mlp->d_dhidden, mlp->d_hidden,
                                              mlp->d_dhidden, hidden_total);

    // 5. Compute dW1 and db1
    dim3 gridDim3((cfg.hidden_size + blockDim.x - 1) / blockDim.x,
                  (cfg.input_size + blockDim.y - 1) / blockDim.y);
    matmul_transpose_kernel<<<gridDim3, blockDim>>>(d_input, mlp->d_dhidden,
                                                    mlp->d_dW1,
                                                    cfg.input_size, cfg.hidden_size,
                                                    batch_size, true, false);

    blocks_1d = (cfg.hidden_size + threads_1d - 1) / threads_1d;
    CUDA_CHECK(cudaMemset(mlp->d_db1, 0, cfg.hidden_size * sizeof(float)));
    sum_columns_kernel<<<blocks_1d, threads_1d>>>(mlp->d_dhidden, mlp->d_db1,
                                            batch_size, cfg.hidden_size);
}

void mlp_update_weights_cuda(MLPCuda *mlp, int batch_size) {
    MLPConfig cfg = mlp->config;
    int threadsPerBlock = 256;
    
    // Update W1
    int numBlocks = (cfg.input_size * cfg.hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(mlp->d_W1, mlp->d_dW1,
                                                      cfg.learning_rate,
                                                      cfg.input_size * cfg.hidden_size,
                                                      batch_size);

    numBlocks = (cfg.hidden_size + threadsPerBlock - 1) / threadsPerBlock;
    sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(mlp->d_b1, mlp->d_db1,
                                                      cfg.learning_rate,
                                                      cfg.hidden_size,
                                                      batch_size);

    numBlocks = (cfg.hidden_size * cfg.output_size + threadsPerBlock - 1) / threadsPerBlock;
    sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(mlp->d_W2, mlp->d_dW2,
                                                      cfg.learning_rate,
                                                      cfg.hidden_size * cfg.output_size,
                                                      batch_size);

    numBlocks = (cfg.output_size + threadsPerBlock - 1) / threadsPerBlock;
    sgd_update_kernel<<<numBlocks, threadsPerBlock>>>(mlp->d_b2, mlp->d_db2,
                                                      cfg.learning_rate,
                                                      cfg.output_size,
                                                      batch_size);
}

float mlp_compute_loss_cuda(MLPCuda *mlp, int *d_labels, int batch_size) {
    MLPConfig cfg = mlp->config;

    if (!mlp->d_temp) {
        CUDA_CHECK(cudaMalloc(&mlp->d_temp, batch_size * cfg.output_size * sizeof(float)));
    }

    float *d_loss = NULL;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    cross_entropy_loss_kernel<<<blocks, threads>>>(mlp->d_output, d_labels,
                                                   d_loss,
                                                   batch_size, cfg.output_size);

    float loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return loss / batch_size;
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
            int offset = batch * batch_size;
            float *batch_data = &train_data[offset * mlp->config.input_size];
            int *batch_labels = &train_labels[offset];

            CUDA_CHECK(cudaMemcpy(mlp->d_input, batch_data,
                                  batch_size * mlp->config.input_size * sizeof(float),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(mlp->d_labels, batch_labels,
                                  batch_size * sizeof(int),
                                  cudaMemcpyHostToDevice));

            mlp_forward_cuda(mlp, mlp->d_input, batch_size);
            float loss = mlp_compute_loss_cuda(mlp, mlp->d_labels, batch_size);
            total_loss += loss;
            mlp_backward_cuda(mlp, mlp->d_input, mlp->d_labels, batch_size);
            mlp_update_weights_cuda(mlp, batch_size);
        }
        
        double epoch_time = timer_stop(&timer);
        printf("Epoch %d/%d - Loss: %.4f - Time: %.2f ms\n",
               epoch + 1, epochs, total_loss / num_batches, epoch_time);
    }
}

float mlp_evaluate_cuda(MLPCuda *mlp, float *test_data, int *test_labels, int num_samples) {
    int batch_size = 256;
    if (num_samples < batch_size) {
        batch_size = num_samples;
    }

    if (!mlp->d_input) {
        CUDA_CHECK(cudaMalloc(&mlp->d_input, batch_size * mlp->config.input_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&mlp->d_labels, batch_size * sizeof(int)));
    }

    int *d_correct = NULL;
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));

    int total_correct = 0;
    int num_batches = (num_samples + batch_size - 1) / batch_size;

    for (int batch = 0; batch < num_batches; batch++) {
        int current_batch = batch_size;
        int offset = batch * batch_size;
        if (offset + current_batch > num_samples) {
            current_batch = num_samples - offset;
        }

        CUDA_CHECK(cudaMemcpy(mlp->d_input, &test_data[offset * mlp->config.input_size],
                              current_batch * mlp->config.input_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mlp->d_labels, &test_labels[offset],
                              current_batch * sizeof(int),
                              cudaMemcpyHostToDevice));

        mlp_forward_cuda(mlp, mlp->d_input, current_batch);

        CUDA_CHECK(cudaMemset(d_correct, 0, sizeof(int)));
        int threads = 256;
        int blocks = (current_batch + threads - 1) / threads;
        compute_accuracy_kernel<<<blocks, threads>>>(mlp->d_output, mlp->d_labels,
                                                     d_correct, current_batch,
                                                     mlp->config.output_size);

        int batch_correct = 0;
        CUDA_CHECK(cudaMemcpy(&batch_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
        total_correct += batch_correct;
    }

    cudaFree(d_correct);
    return (float)total_correct / num_samples;
}

#include "mlp.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Create MLP on CPU
MLP* mlp_create_cpu(MLPConfig config) {
    MLP *mlp = (MLP*)malloc(sizeof(MLP));
    mlp->config = config;
    
    // Allocate weights and biases
    mlp->W1 = (float*)malloc(config.input_size * config.hidden_size * sizeof(float));
    mlp->b1 = (float*)malloc(config.hidden_size * sizeof(float));
    mlp->W2 = (float*)malloc(config.hidden_size * config.output_size * sizeof(float));
    mlp->b2 = (float*)malloc(config.output_size * sizeof(float));
    
    // Allocate intermediate buffers (sized for max batch)
    mlp->hidden = NULL;  
    mlp->output = NULL;
    
    // Allocate gradients
    mlp->dW1 = (float*)calloc(config.input_size * config.hidden_size, sizeof(float));
    mlp->db1 = (float*)calloc(config.hidden_size, sizeof(float));
    mlp->dW2 = (float*)calloc(config.hidden_size * config.output_size, sizeof(float));
    mlp->db2 = (float*)calloc(config.output_size, sizeof(float));
    mlp->dhidden = NULL;
    
    // Initialize weights randomly
    initialize_weights_random(mlp->W1, config.input_size * config.hidden_size, 
                             sqrt(2.0 / config.input_size));
    initialize_weights_random(mlp->W2, config.hidden_size * config.output_size,
                             sqrt(2.0 / config.hidden_size));
    initialize_bias_zero(mlp->b1, config.hidden_size);
    initialize_bias_zero(mlp->b2, config.output_size);
    
    return mlp;
}

// Free CPU memory and destroy MLP
void mlp_destroy_cpu(MLP *mlp) {
    // Free weights, biases, intermediate values, and gradients
    free(mlp->W1);
    free(mlp->b1);
    free(mlp->W2);
    free(mlp->b2);
    if (mlp->hidden) free(mlp->hidden);
    if (mlp->output) free(mlp->output);
    if (mlp->dhidden) free(mlp->dhidden);
    free(mlp->dW1);
    free(mlp->db1);
    free(mlp->dW2);
    free(mlp->db2);
    free(mlp);
}

// ==================== CPU IMPLEMENTATION ====================

// Forward pass on CPU
void mlp_forward_cpu(MLP *mlp, float *input, int batch_size) {
    // Allocate intermediate buffers if needed
    if (!mlp->hidden) {
        mlp->hidden = (float*)malloc(batch_size * mlp->config.hidden_size * sizeof(float));
        mlp->output = (float*)malloc(batch_size * mlp->config.output_size * sizeof(float));
        mlp->dhidden = (float*)malloc(batch_size * mlp->config.hidden_size * sizeof(float));
    }
    
    // Layer 1: hidden = input @ W1 + b1
    // input: [batch_size, input_size], W1: [input_size, hidden_size]
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < mlp->config.hidden_size; h++) {
            float sum = 0.0f;
            for (int i = 0; i < mlp->config.input_size; i++) {
                sum += input[b * mlp->config.input_size + i] * 
                       mlp->W1[i * mlp->config.hidden_size + h];
            }
            mlp->hidden[b * mlp->config.hidden_size + h] = sum + mlp->b1[h];
        }
    }
    
    // ReLU activation
    for (int i = 0; i < batch_size * mlp->config.hidden_size; i++) {
        mlp->hidden[i] = fmaxf(0.0f, mlp->hidden[i]);
    }
    
    // Layer 2: output = hidden @ W2 + b2
    // hidden: [batch_size, hidden_size], W2: [hidden_size, output_size]
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < mlp->config.output_size; o++) {
            float sum = 0.0f;
            for (int h = 0; h < mlp->config.hidden_size; h++) {
                sum += mlp->hidden[b * mlp->config.hidden_size + h] * 
                       mlp->W2[h * mlp->config.output_size + o];
            }
            mlp->output[b * mlp->config.output_size + o] = sum + mlp->b2[o];
        }
    }
    
    // Softmax activation
    for (int b = 0; b < batch_size; b++) {
        float *output_row = &mlp->output[b * mlp->config.output_size];
        
        // Find max for numerical stability
        float max_val = output_row[0];
        for (int i = 1; i < mlp->config.output_size; i++) {
            if (output_row[i] > max_val) max_val = output_row[i];
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < mlp->config.output_size; i++) {
            output_row[i] = expf(output_row[i] - max_val);
            sum += output_row[i];
        }
        
        // Normalize
        for (int i = 0; i < mlp->config.output_size; i++) {
            output_row[i] /= sum;
        }
    }
}

// Backward pass on CPU
void mlp_backward_cpu(MLP *mlp, float *input, int *labels, int batch_size) {
    // Compute output gradient: dL/dz = output - one_hot(labels)
    float *doutput = (float*)malloc(batch_size * mlp->config.output_size * sizeof(float));
    
    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < mlp->config.output_size; o++) {
            doutput[b * mlp->config.output_size + o] = mlp->output[b * mlp->config.output_size + o];
            if (o == labels[b]) {
                doutput[b * mlp->config.output_size + o] -= 1.0f;
            }
        }
    }
    
    // Backprop through layer 2: dW2 = hidden^T @ doutput, db2 = sum(doutput)
    // dW2: [hidden_size, output_size]
    for (int h = 0; h < mlp->config.hidden_size; h++) {
        for (int o = 0; o < mlp->config.output_size; o++) {
            float grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                grad += mlp->hidden[b * mlp->config.hidden_size + h] * 
                        doutput[b * mlp->config.output_size + o];
            }
            mlp->dW2[h * mlp->config.output_size + o] += grad;
        }
    }
    
    for (int o = 0; o < mlp->config.output_size; o++) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += doutput[b * mlp->config.output_size + o];
        }
        mlp->db2[o] += grad;
    }
    
    // Backprop to hidden: dhidden = doutput @ W2^T
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < mlp->config.hidden_size; h++) {
            float grad = 0.0f;
            for (int o = 0; o < mlp->config.output_size; o++) {
                grad += doutput[b * mlp->config.output_size + o] * 
                        mlp->W2[h * mlp->config.output_size + o];
            }
            mlp->dhidden[b * mlp->config.hidden_size + h] = grad;
        }
    }
    
    // ReLU backward
    for (int i = 0; i < batch_size * mlp->config.hidden_size; i++) {
        if (mlp->hidden[i] <= 0.0f) {
            mlp->dhidden[i] = 0.0f;
        }
    }
    
    // Backprop through layer 1: dW1 = input^T @ dhidden, db1 = sum(dhidden)
    for (int i = 0; i < mlp->config.input_size; i++) {
        for (int h = 0; h < mlp->config.hidden_size; h++) {
            float grad = 0.0f;
            for (int b = 0; b < batch_size; b++) {
                grad += input[b * mlp->config.input_size + i] * 
                        mlp->dhidden[b * mlp->config.hidden_size + h];
            }
            mlp->dW1[i * mlp->config.hidden_size + h] += grad;
        }
    }
    
    for (int h = 0; h < mlp->config.hidden_size; h++) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += mlp->dhidden[b * mlp->config.hidden_size + h];
        }
        mlp->db1[h] += grad;
    }
    
    free(doutput);
}

// Update weights using SGD
void mlp_update_weights_cpu(MLP *mlp) {
    float lr = mlp->config.learning_rate;
    
    // Update W1 and b1
    for (int i = 0; i < mlp->config.input_size * mlp->config.hidden_size; i++) {
        mlp->W1[i] -= lr * mlp->dW1[i];
        mlp->dW1[i] = 0.0f;  // Reset gradient
    }
    
    for (int i = 0; i < mlp->config.hidden_size; i++) {
        mlp->b1[i] -= lr * mlp->db1[i];
        mlp->db1[i] = 0.0f;
    }
    
    // Update W2 and b2
    for (int i = 0; i < mlp->config.hidden_size * mlp->config.output_size; i++) {
        mlp->W2[i] -= lr * mlp->dW2[i];
        mlp->dW2[i] = 0.0f;
    }
    
    for (int i = 0; i < mlp->config.output_size; i++) {
        mlp->b2[i] -= lr * mlp->db2[i];
        mlp->db2[i] = 0.0f;
    }
}

// Compute cross-entropy loss
float mlp_compute_loss_cpu(MLP *mlp, int *labels, int batch_size) {
    float loss = 0.0f;
    
    for (int b = 0; b < batch_size; b++) {
        int label = labels[b];
        float prob = mlp->output[b * mlp->config.output_size + label];
        // Cross-entropy: -log(prob)
        loss -= logf(fmaxf(prob, 1e-10f));  // Avoid log(0)
    }
    
    return loss / batch_size;
}

// Predict the class for a single input sample
int mlp_predict_cpu(MLP *mlp, float *input) {
    mlp_forward_cpu(mlp, input, 1);
    
    // Find argmax
    int prediction = 0;
    float max_val = mlp->output[0];
    for (int i = 1; i < mlp->config.output_size; i++) {
        if (mlp->output[i] > max_val) {
            max_val = mlp->output[i];
            prediction = i;
        }
    }
    return prediction;
}

// Train the MLP on CPU
void mlp_train_cpu(MLP *mlp, float *train_data, int *train_labels,
                   int num_samples, int batch_size, int epochs) {
    printf("Training MLP on CPU...\n");
    Timer timer;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        timer_start(&timer);
        float total_loss = 0.0f;
        int num_batches = num_samples / batch_size;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int offset = batch * batch_size;
            float *batch_data = &train_data[offset * mlp->config.input_size];
            int *batch_labels = &train_labels[offset];
            
            // Forward pass
            mlp_forward_cpu(mlp, batch_data, batch_size);
            
            // Compute loss
            float loss = mlp_compute_loss_cpu(mlp, batch_labels, batch_size);
            total_loss += loss;
            
            // Backward pass
            mlp_backward_cpu(mlp, batch_data, batch_labels, batch_size);
            
            // Update weights
            mlp_update_weights_cpu(mlp);
        }
        
        double epoch_time = timer_stop(&timer);
        printf("Epoch %d/%d - Loss: %.4f - Time: %.2f ms\n", 
               epoch + 1, epochs, total_loss / num_batches, epoch_time);
    }
}

// Evaluate the MLP on CPU
float mlp_evaluate_cpu(MLP *mlp, float *test_data, int *test_labels, int num_samples) {
    int correct = 0;
    
    for (int i = 0; i < num_samples; i++) {
        int prediction = mlp_predict_cpu(mlp, &test_data[i * mlp->config.input_size]);
        if (prediction == test_labels[i]) {
            correct++;
        }
    }
    
    return (float)correct / num_samples;
}

#ifndef MLP_H
#define MLP_H

#include <stddef.h>

// MLP Network Configuration
typedef struct {
    int input_size;      // 784 for MNIST (28x28)
    int hidden_size;     // Number of hidden neurons
    int output_size;     // 10 for digits 0-9
    float learning_rate; // SGD learning rate
} MLPConfig;

// MLP Network Structure
typedef struct {
    MLPConfig config;
    
    // CPU Memory
    float *W1;           // Input to hidden weights
    float *b1;           // Hidden layer bias
    float *W2;           // Hidden to output weights
    float *b2;           // Output layer bias
    
    // Forward pass intermediate values
    float *hidden;       // Hidden layer activations
    float *output;       // Output layer (softmax)
    
    // Gradients for backpropagation
    float *dW1, *db1;
    float *dW2, *db2;
    float *dhidden;
} MLP;

// CPU Implementation Functions
MLP* mlp_create_cpu(MLPConfig config);
void mlp_destroy_cpu(MLP *mlp);
void mlp_forward_cpu(MLP *mlp, float *input, int batch_size);
void mlp_backward_cpu(MLP *mlp, float *input, int *labels, int batch_size);
void mlp_update_weights_cpu(MLP *mlp);
float mlp_compute_loss_cpu(MLP *mlp, int *labels, int batch_size);
int mlp_predict_cpu(MLP *mlp, float *input);

// Training utilities
void mlp_train_cpu(MLP *mlp, float *train_data, int *train_labels, 
                   int num_samples, int batch_size, int epochs);
float mlp_evaluate_cpu(MLP *mlp, float *test_data, int *test_labels, int num_samples);

#endif // MLP_H

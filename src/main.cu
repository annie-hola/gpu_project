#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlp.h"
#include "mlp_cuda.h"
#include "mnist_loader.h"
#include "utils.h"

// Configuration
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01f
#define BATCH_SIZE 64
#define EPOCHS 10

void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Options:\n");
    printf("  --mode <cpu|gpu|both>     Training mode (default: both)\n");
    printf("  --epochs <N>              Number of training epochs (default: 10)\n");
    printf("  --batch-size <N>          Batch size (default: 64)\n");
    printf("  --hidden-size <N>         Hidden layer size (default: 256)\n");
    printf("  --learning-rate <F>       Learning rate (default: 0.01)\n");
    printf("  --data-dir <PATH>         Path to MNIST data (default: ./data)\n");
    printf("  --help                    Show this help message\n");
}

int main(int argc, char **argv) {
    // Parse command line arguments
    char *mode = "both";
    int epochs = EPOCHS;
    int batch_size = BATCH_SIZE;
    int hidden_size = HIDDEN_SIZE;
    float learning_rate = LEARNING_RATE;
    char *data_dir = "data";
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden-size") == 0 && i + 1 < argc) {
            hidden_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    printf("=== MNIST Digit Classification with MLP ===\n");
    printf("Configuration:\n");
    printf("  Mode: %s\n", mode);
    printf("  Epochs: %d\n", epochs);
    printf("  Batch Size: %d\n", batch_size);
    printf("  Hidden Size: %d\n", hidden_size);
    printf("  Learning Rate: %.4f\n", learning_rate);
    printf("  Data Directory: %s\n\n", data_dir);
    
    // Print GPU information if using GPU
    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0) {
        cuda_print_device_info();
    }
    
    // Load MNIST dataset
    printf("Loading MNIST dataset...\n");
    
    char train_images_path[256], train_labels_path[256];
    char test_images_path[256], test_labels_path[256];
    
    snprintf(train_images_path, sizeof(train_images_path), "%s/train-images-idx3-ubyte", data_dir);
    snprintf(train_labels_path, sizeof(train_labels_path), "%s/train-labels-idx1-ubyte", data_dir);
    snprintf(test_images_path, sizeof(test_images_path), "%s/t10k-images-idx3-ubyte", data_dir);
    snprintf(test_labels_path, sizeof(test_labels_path), "%s/t10k-labels-idx1-ubyte", data_dir);
    
    MNISTDataset *train_data = load_mnist_train(train_images_path, train_labels_path);
    MNISTDataset *test_data = load_mnist_test(test_images_path, test_labels_path);
    
    if (!train_data || !test_data) {
        fprintf(stderr, "Failed to load MNIST dataset.\n");
        fprintf(stderr, "Please ensure MNIST files are in the %s directory.\n", data_dir);
        return 1;
    }
    
    printf("Training samples: %d\n", train_data->num_images);
    printf("Test samples: %d\n\n", test_data->num_images);
    
    // Create MLP configuration
    MLPConfig config = {
        .input_size = INPUT_SIZE,
        .hidden_size = hidden_size,
        .output_size = OUTPUT_SIZE,
        .learning_rate = learning_rate
    };
    
    // ==================== CPU TRAINING ====================
    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "both") == 0) {
        printf("\n========== CPU Training ==========\n");
        
        MLP *mlp_cpu = mlp_create_cpu(config);
        
        Timer timer;
        timer_start(&timer);
        
        mlp_train_cpu(mlp_cpu, train_data->images, train_data->labels,
                     train_data->num_images, batch_size, epochs);
        
        double total_time = timer_stop(&timer);
        printf("Total CPU training time: %.2f ms (%.2f s)\n", total_time, total_time / 1000.0);
        
        // Evaluate on test set
        printf("\nEvaluating on test set...\n");
        float cpu_accuracy = mlp_evaluate_cpu(mlp_cpu, test_data->images, 
                                              test_data->labels, test_data->num_images);
        printf("CPU Test Accuracy: %.2f%%\n", cpu_accuracy * 100);
        
        mlp_destroy_cpu(mlp_cpu);
    }
    
    // ==================== GPU TRAINING ====================
    if (strcmp(mode, "gpu") == 0 || strcmp(mode, "both") == 0) {
        printf("\n========== GPU Training ==========\n");
        
        MLPCuda *mlp_gpu = mlp_create_cuda(config);
        
        // Initialize weights (same as CPU for fair comparison)
        if (strcmp(mode, "both") == 0) {
            MLP *mlp_temp = mlp_create_cpu(config);
            mlp_copy_weights_to_device(mlp_gpu, mlp_temp);
            mlp_destroy_cpu(mlp_temp);
        } else {
            // Create temporary CPU model just for initialization
            MLP *mlp_temp = mlp_create_cpu(config);
            mlp_copy_weights_to_device(mlp_gpu, mlp_temp);
            mlp_destroy_cpu(mlp_temp);
        }
        
        Timer timer;
        timer_start(&timer);
        
        mlp_train_cuda(mlp_gpu, train_data->images, train_data->labels,
                      train_data->num_images, batch_size, epochs);
        
        double total_time = timer_stop(&timer);
        printf("Total GPU training time: %.2f ms (%.2f s)\n", total_time, total_time / 1000.0);
        
        // Evaluate on test set
        printf("\nEvaluating on test set...\n");
        float gpu_accuracy = mlp_evaluate_cuda(mlp_gpu, test_data->images,
                                               test_data->labels, test_data->num_images);
        printf("GPU Test Accuracy: %.2f%%\n", gpu_accuracy * 100);
        
        mlp_destroy_cuda(mlp_gpu);
    }
    
    // ==================== PERFORMANCE COMPARISON ====================
    if (strcmp(mode, "both") == 0) {
        printf("\n========== Performance Summary ==========\n");
        printf("CPU vs GPU speedup calculation requires full implementation.\n");
        printf("Metrics to compare:\n");
        printf("  - Training time per epoch\n");
        printf("  - Total training time\n");
        printf("  - Accuracy convergence\n");
        printf("  - Memory usage\n");
    }
    
    // Cleanup
    free_mnist_dataset(train_data);
    free_mnist_dataset(test_data);
    
    printf("\nTraining completed successfully!\n");
    return 0;
}

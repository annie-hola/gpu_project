#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ==================== TIMER UTILITIES ====================

void timer_start(Timer *timer) {
    gettimeofday(&timer->start, NULL);
}

double timer_stop(Timer *timer) {
    gettimeofday(&timer->end, NULL);
    double start_ms = timer->start.tv_sec * 1000.0 + timer->start.tv_usec / 1000.0;
    double end_ms = timer->end.tv_sec * 1000.0 + timer->end.tv_usec / 1000.0;
    return end_ms - start_ms;
}

// ==================== CUDA UTILITIES ====================

void cuda_check_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cuda_print_device_info() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    printf("=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
        printf("  Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    }
    printf("===============================\n\n");
}

// ==================== RANDOM INITIALIZATION ====================

void initialize_weights_random(float *weights, int size, float scale) {
    // Use normal distribution with mean=0, std=scale
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(time(NULL));
        seed_initialized = 1;
    }
    
    for (int i = 0; i < size; i++) {
        // Box-Muller transform for normal distribution
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        weights[i] = z * scale;
    }
}

void initialize_bias_zero(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// ==================== MATH UTILITIES ====================

float compute_accuracy(int *predictions, int *labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return (float)correct / num_samples;
}

void print_progress_bar(int current, int total, double loss, float accuracy) {
    int bar_width = 50;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d/%d (%.1f%%) - Loss: %.4f - Acc: %.2f%%",
           current, total, progress * 100, loss, accuracy * 100);
    fflush(stdout);
    
    if (current == total) printf("\n");
}

// ==================== MEMORY UTILITIES ====================

size_t get_available_gpu_memory() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem;
}

void print_memory_usage(const char *label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    printf("[%s] GPU Memory - Used: %.2f MB / Total: %.2f MB (%.1f%%)\n",
           label,
           used_mem / 1e6,
           total_mem / 1e6,
           (float)used_mem / total_mem * 100);
}

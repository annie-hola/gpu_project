#ifndef UTILS_H
#define UTILS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <sys/time.h>

// Timer utilities
typedef struct {
    struct timeval start;
    struct timeval end;
} Timer;

void timer_start(Timer *timer);
double timer_stop(Timer *timer);  // Returns elapsed time in milliseconds

#ifdef __CUDACC__
// CUDA utilities
void cuda_check_error(cudaError_t err, const char *file, int line);
void cuda_print_device_info();

#define CUDA_CHECK(err) cuda_check_error(err, __FILE__, __LINE__)

// Memory utilities
size_t get_available_gpu_memory();
void print_memory_usage(const char *label);
#endif

// Random initialization
void initialize_weights_random(float *weights, int size, float scale);
void initialize_bias_zero(float *bias, int size);

// Math utilities
float compute_accuracy(int *predictions, int *labels, int num_samples);
void print_progress_bar(int current, int total, double loss, float accuracy);

#endif // UTILS_H

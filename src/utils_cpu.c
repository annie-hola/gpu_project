#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "utils.h"

// Timer implementation
void timer_start(Timer *timer) {
    gettimeofday(&timer->start, NULL);
}

double timer_stop(Timer *timer) {
    gettimeofday(&timer->end, NULL);
    double start_time = (double)timer->start.tv_sec * 1000.0 + (double)timer->start.tv_usec / 1000.0;
    double end_time = (double)timer->end.tv_sec * 1000.0 + (double)timer->end.tv_usec / 1000.0;
    return end_time - start_time;
}

// Random weight initialization (He initialization)
void initialize_weights_random(float *weights, int size, float scale) {
    for (int i = 0; i < size; i++) {
        // Box-Muller transform for Gaussian random numbers
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z0 = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(2.0f * 3.14159265f * u2);
        weights[i] = z0 * scale;
    }
}

// Zero initialization for biases
void initialize_bias_zero(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// Compute accuracy
float compute_accuracy(int *predictions, int *labels, int num_samples) {
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        if (predictions[i] == labels[i]) {
            correct++;
        }
    }
    return (float)correct / (float)num_samples * 100.0f;
}

// Print progress bar
void print_progress_bar(int current, int total, double loss, float accuracy) {
    int bar_width = 30;
    float progress = (float)current / (float)total;
    int filled = (int)(progress * bar_width);
    
    printf("\r[");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else printf(" ");
    }
    printf("] %.1f%% - Loss: %.4f - Acc: %.2f%%", progress * 100.0f, loss, accuracy);
    fflush(stdout);
    
    if (current == total) printf("\n");
}

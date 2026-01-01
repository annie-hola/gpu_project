#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <stddef.h>

// MNIST dataset structure
typedef struct {
    float *images;       // Flattened images (num_images x 784)
    int *labels;         // Labels (num_images)
    int num_images;      // Total number of images
    int image_size;      // 784 (28x28)
} MNISTDataset;

// Load MNIST dataset from files
MNISTDataset* load_mnist_train(const char *images_path, const char *labels_path);
MNISTDataset* load_mnist_test(const char *images_path, const char *labels_path);

// Free dataset memory
void free_mnist_dataset(MNISTDataset *dataset);

// Utility functions
void normalize_mnist_images(float *images, int num_images, int image_size);
void shuffle_dataset(float *images, int *labels, int num_samples, int image_size);

// Download MNIST if not present
int download_mnist_dataset(const char *data_dir);

#endif // MNIST_LOADER_H

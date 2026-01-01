#include "mnist_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Helper function to read big-endian 32-bit integer
static uint32_t read_uint32(FILE *fp) {
    uint32_t value;
    fread(&value, sizeof(uint32_t), 1, fp);
    // Convert from big-endian to host byte order
    return __builtin_bswap32(value);
}

MNISTDataset* load_mnist_train(const char *images_path, const char *labels_path) {
    MNISTDataset *dataset = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    
    // Load images
    FILE *images_file = fopen(images_path, "rb");
    if (!images_file) {
        fprintf(stderr, "Error: Cannot open images file: %s\n", images_path);
        free(dataset);
        return NULL;
    }
    
    // Read magic number and dimensions
    uint32_t magic = read_uint32(images_file);
    if (magic != 0x00000803) {
        fprintf(stderr, "Error: Invalid magic number in images file\n");
        fclose(images_file);
        free(dataset);
        return NULL;
    }
    
    uint32_t num_images = read_uint32(images_file);
    uint32_t num_rows = read_uint32(images_file);
    uint32_t num_cols = read_uint32(images_file);
    
    dataset->num_images = num_images;
    dataset->image_size = num_rows * num_cols;
    
    // Allocate and read image data
    dataset->images = (float*)malloc(num_images * dataset->image_size * sizeof(float));
    uint8_t *buffer = (uint8_t*)malloc(dataset->image_size);
    
    for (int i = 0; i < num_images; i++) {
        fread(buffer, 1, dataset->image_size, images_file);
        for (int j = 0; j < dataset->image_size; j++) {
            dataset->images[i * dataset->image_size + j] = (float)buffer[j];
        }
    }
    
    free(buffer);
    fclose(images_file);
    
    // Load labels
    FILE *labels_file = fopen(labels_path, "rb");
    if (!labels_file) {
        fprintf(stderr, "Error: Cannot open labels file: %s\n", labels_path);
        free(dataset->images);
        free(dataset);
        return NULL;
    }
    
    magic = read_uint32(labels_file);
    if (magic != 0x00000801) {
        fprintf(stderr, "Error: Invalid magic number in labels file\n");
        fclose(labels_file);
        free(dataset->images);
        free(dataset);
        return NULL;
    }
    
    uint32_t num_labels = read_uint32(labels_file);
    dataset->labels = (int*)malloc(num_labels * sizeof(int));
    
    uint8_t *label_buffer = (uint8_t*)malloc(num_labels);
    fread(label_buffer, 1, num_labels, labels_file);
    
    for (int i = 0; i < num_labels; i++) {
        dataset->labels[i] = (int)label_buffer[i];
    }
    
    free(label_buffer);
    fclose(labels_file);
    
    // Normalize images to [0, 1]
    normalize_mnist_images(dataset->images, dataset->num_images, dataset->image_size);
    
    printf("Loaded %d training images\n", dataset->num_images);
    return dataset;
}

MNISTDataset* load_mnist_test(const char *images_path, const char *labels_path) {
    return load_mnist_train(images_path, labels_path);
}

void free_mnist_dataset(MNISTDataset *dataset) {
    if (dataset) {
        if (dataset->images) free(dataset->images);
        if (dataset->labels) free(dataset->labels);
        free(dataset);
    }
}

void normalize_mnist_images(float *images, int num_images, int image_size) {
    int total_pixels = num_images * image_size;
    for (int i = 0; i < total_pixels; i++) {
        images[i] /= 255.0f;
    }
}

void shuffle_dataset(float *images, int *labels, int num_samples, int image_size) {
    for (int i = num_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        
        // Swap labels
        int temp_label = labels[i];
        labels[i] = labels[j];
        labels[j] = temp_label;
        
        // Swap images
        for (int k = 0; k < image_size; k++) {
            float temp = images[i * image_size + k];
            images[i * image_size + k] = images[j * image_size + k];
            images[j * image_size + k] = temp;
        }
    }
}

int download_mnist_dataset(const char *data_dir) {
    printf("Auto-download not implemented yet.\n");
    printf("Please manually download MNIST dataset to: %s\n", data_dir);
    printf("Visit: http://yann.lecun.com/exdb/mnist/\n");
    return -1;
}

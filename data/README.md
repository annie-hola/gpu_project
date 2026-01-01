# MNIST Data Directory

This directory will contain the MNIST dataset files after downloading.

## Download Instructions

Run the following command from the project root:

```bash
make download-data
```

This will download and extract the following files:
- `train-images-idx3-ubyte` - Training set images (60,000 samples)
- `train-labels-idx1-ubyte` - Training set labels
- `t10k-images-idx3-ubyte` - Test set images (10,000 samples)
- `t10k-labels-idx1-ubyte` - Test set labels


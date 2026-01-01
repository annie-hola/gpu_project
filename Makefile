# MNIST MLP Training - CPU Only
# Makefile for building CPU-only version (no CUDA required)

# Compiler and flags
CC = gcc
CXX = g++

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin
DATA_DIR = data

# Compiler flags (CPU)
CC_FLAGS = -O3 -Wall -I$(INC_DIR)

# Target executable
TARGET = $(BIN_DIR)/mnist_mlp

# Source files (CPU-only)
C_SOURCES = $(SRC_DIR)/main.cu \
            $(SRC_DIR)/mlp.c \
            $(SRC_DIR)/mnist_loader.c \
            $(SRC_DIR)/utils_cpu.c

# Object files
C_OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.o, $(filter %.cu, $(C_SOURCES))) \
            $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(filter %.c, $(C_SOURCES)))

# Default target
.PHONY: all
all: directories $(TARGET)

# Create necessary directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(DATA_DIR)

# Link all object files
$(TARGET): $(C_OBJECTS)
	@echo "Linking $@..."
	$(CC) -o $@ $^ $(CC_FLAGS) -lm
	@echo "Build complete: $@"

# Compile .cu files as C (ignoring CUDA code)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $< (as C)..."
	$(CC) -x c $(CC_FLAGS) -c $< -o $@

# Compile C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CC_FLAGS) -c $< -o $@

# Run the program with default settings
.PHONY: run
run: all
	@echo "Running MNIST MLP training (CPU)..."
	./$(TARGET) --mode cpu --epochs 10 --batch-size 64

# Run with custom epochs
.PHONY: run-train
run-train: all
	@echo "Training MLP..."
	./$(TARGET) --mode cpu --epochs 5 --batch-size 128

# Download MNIST dataset (requires wget or curl)
.PHONY: download-data
download-data:
	@echo "Downloading MNIST dataset..."
	@mkdir -p $(DATA_DIR)
	@if command -v curl > /dev/null; then \
		curl -L -o $(DATA_DIR)/train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz; \
		curl -L -o $(DATA_DIR)/train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz; \
		curl -L -o $(DATA_DIR)/t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz; \
		curl -L -o $(DATA_DIR)/t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz; \
	elif command -v wget > /dev/null; then \
		wget -P $(DATA_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz; \
		wget -P $(DATA_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz; \
		wget -P $(DATA_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz; \
		wget -P $(DATA_DIR) https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz; \
	else \
		echo "Error: wget or curl not found. Please install one of them."; \
		exit 1; \
	fi
	@echo "Extracting files..."
	@gunzip -f $(DATA_DIR)/*.gz
	@echo "MNIST dataset downloaded and extracted to $(DATA_DIR)/"

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)/*.o
	@rm -f $(TARGET)
	@echo "Clean complete."

# Deep clean (including data)
.PHONY: distclean
distclean: clean
	@echo "Deep cleaning..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@echo "Deep clean complete."

# Display help
.PHONY: help
help:
	@echo "MNIST MLP CPU-Only Project - Makefile Help"
	@echo ""
	@echo "Available targets:"
	@echo "  make                 - Build CPU-only version"
	@echo "  make all             - Build CPU-only version"
	@echo "  make run             - Build and run with CPU (10 epochs)"
	@echo "  make run-train       - Build and run training (5 epochs)"
	@echo "  make download-data   - Download MNIST dataset"
	@echo "  make clean           - Remove build artifacts"
	@echo "  make distclean       - Remove all generated files"
	@echo "  make help            - Display this help message"
	@echo ""
	@echo "Usage: make -f Makefile.cpu [target]"
	@echo "Example: make -f Makefile.cpu run"

# Dependencies (header files)
$(BUILD_DIR)/main.o: $(INC_DIR)/mlp.h $(INC_DIR)/mlp_cuda.h $(INC_DIR)/mnist_loader.h $(INC_DIR)/utils.h
$(BUILD_DIR)/mlp.o: $(INC_DIR)/mlp.h $(INC_DIR)/utils.h
$(BUILD_DIR)/mnist_loader.o: $(INC_DIR)/mnist_loader.h
$(BUILD_DIR)/utils_cpu.o: $(INC_DIR)/utils.h

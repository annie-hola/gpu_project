#!/bin/bash

# ==================== GPU OPTIMIZATION EXPERIMENTS ====================
# This script runs systematic experiments varying GPU constants and hyperparameters

RESULTS_DIR="result/experiments"
mkdir -p "$RESULTS_DIR"

echo "======================================================"
echo "GPU OPTIMIZATION EXPERIMENTS"
echo "======================================================"
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# ==================== EXPERIMENT 1: THREADS_PER_BLOCK_1D ====================
echo "EXPERIMENT 1: Varying THREADS_PER_BLOCK_1D (1D Kernel Thread Count)"
echo "Default: 256 threads"
echo "Testing: 128, 256, 512"
echo ""

for threads in 128 256 512; do
    echo ">>> Testing THREADS_PER_BLOCK_1D = $threads"
    
    # Modify cuda_constants.h
    sed -i.bak "s/#define THREADS_PER_BLOCK_1D ..*/#define THREADS_PER_BLOCK_1D $threads/" include/cuda_constants.h
    
    # Rebuild
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    
    if [ ! -f "bin/mnist_mlp" ]; then
        echo "Build failed for THREADS_PER_BLOCK_1D=$threads"
        continue
    fi
    
    # Run experiment
    output_file="$RESULTS_DIR/exp1_threads1d_${threads}.txt"
    ./bin/mnist_mlp --mode both --epochs 5 --batch-size 64 > "$output_file" 2>&1
    
    echo "Results saved to $output_file"
    echo ""
done

# Restore default
sed -i "s/#define THREADS_PER_BLOCK_1D ..*/#define THREADS_PER_BLOCK_1D 256/" include/cuda_constants.h

# ==================== EXPERIMENT 2: THREADS_PER_BLOCK_2D ====================
echo "EXPERIMENT 2: Varying THREADS_PER_BLOCK_2D (2D Matrix Operation Block Size)"
echo "Default: 16 (16x16 = 256 threads)"
echo "Testing: 8 (8x8 = 64 threads), 16 (16x16 = 256 threads), 32 (32x32 = 1024 threads)"
echo ""

for block_size in 8 16 32; do
    echo ">>> Testing THREADS_PER_BLOCK_2D = $block_size (${block_size}x${block_size})"
    
    # Modify cuda_constants.h
    sed -i.bak "s/#define THREADS_PER_BLOCK_2D ..*/#define THREADS_PER_BLOCK_2D $block_size/" include/cuda_constants.h
    
    # Rebuild
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    
    if [ ! -f "bin/mnist_mlp" ]; then
        echo "Build failed for THREADS_PER_BLOCK_2D=$block_size"
        continue
    fi
    
    # Run experiment
    output_file="$RESULTS_DIR/exp2_threads2d_${block_size}x${block_size}.txt"
    ./bin/mnist_mlp --mode both --epochs 5 --batch-size 64 > "$output_file" 2>&1
    
    echo "Results saved to $output_file"
    echo ""
done

# Restore default
sed -i "s/#define THREADS_PER_BLOCK_2D ..*/#define THREADS_PER_BLOCK_2D 16/" include/cuda_constants.h

# ==================== EXPERIMENT 3: HIDDEN LAYER SIZE ====================
echo "EXPERIMENT 3: Varying Hidden Layer Size (Network Capacity & Parallelism)"
echo "Default: 256"
echo "Testing: 128, 256, 512, 1024"
echo ""

for hidden_size in 128 256 512 1024; do
    echo ">>> Testing Hidden Size = $hidden_size"
    
    # Rebuild (no constant change needed, hyperparameter is in main.c)
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    
    # Run experiment
    output_file="$RESULTS_DIR/exp3_hidden_${hidden_size}.txt"
    ./bin/mnist_mlp --mode both --hidden-size "$hidden_size" --epochs 5 --batch-size 64 > "$output_file" 2>&1
    
    echo "Results saved to $output_file"
    echo ""
done

# ==================== EXPERIMENT 4: LEARNING RATE ====================
echo "EXPERIMENT 4: Varying Learning Rate (Convergence & Numerical Stability)"
echo "Default: 0.01"
echo "Testing: 0.001, 0.01, 0.1"
echo ""

for lr in 0.001 0.01 0.1; do
    echo ">>> Testing Learning Rate = $lr"
    
    # Run experiment
    output_file="$RESULTS_DIR/exp4_lr_${lr}.txt"
    ./bin/mnist_mlp --mode both --learning-rate "$lr" --epochs 5 --batch-size 64 > "$output_file" 2>&1
    
    echo "Results saved to $output_file"
    echo ""
done

# ==================== EXPERIMENT 5: EVAL_BATCH_SIZE ====================
echo "EXPERIMENT 5: Varying EVAL_BATCH_SIZE (Evaluation Throughput)"
echo "Default: 256"
echo "Testing: 64, 256, 512"
echo ""

for eval_batch in 64 256 512; do
    echo ">>> Testing EVAL_BATCH_SIZE = $eval_batch"
    
    # Modify cuda_constants.h
    sed -i.bak "s/#define EVAL_BATCH_SIZE ..*/#define EVAL_BATCH_SIZE $eval_batch/" include/cuda_constants.h
    
    # Rebuild
    make clean > /dev/null 2>&1
    make > /dev/null 2>&1
    
    if [ ! -f "bin/mnist_mlp" ]; then
        echo "Build failed for EVAL_BATCH_SIZE=$eval_batch"
        continue
    fi
    
    # Run experiment
    output_file="$RESULTS_DIR/exp5_eval_batch_${eval_batch}.txt"
    ./bin/mnist_mlp --mode gpu --epochs 5 --batch-size 64 > "$output_file" 2>&1
    
    echo "Results saved to $output_file"
    echo ""
done

# Restore default
sed -i "s/#define EVAL_BATCH_SIZE ..*/#define EVAL_BATCH_SIZE 256/" include/cuda_constants.h

# ==================== CLEANUP ====================
echo "======================================================"
echo "All experiments completed!"
echo "Results saved in: $RESULTS_DIR"
echo ""

# Restore and rebuild to default state
make clean > /dev/null 2>&1
make > /dev/null 2>&1

rm -f include/cuda_constants.h.bak

echo "Project rebuilt with default settings."
echo "======================================================"

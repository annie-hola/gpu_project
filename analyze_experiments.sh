#!/bin/bash

# Analyze GPU Optimization Experiments Results
set -e

RESULTS_DIR="result/experiments"

echo "=========================================="
echo "GPU Optimization Experiments Analysis"
echo "=========================================="
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR does not exist. Run ./run_experiments.sh first."
    exit 1
fi

# Function to extract GPU time from output file
extract_gpu_time() {
    local file=$1
    grep "GPU time:" "$file" | tail -1 | awk '{print $NF}' | sed 's/s//'
}

# Function to extract accuracy from output file
extract_accuracy() {
    local file=$1
    grep "Accuracy:" "$file" | tail -1 | awk '{print $NF}' | sed 's/%//'
}

# Function to extract CPU time from output file (for speedup calculation)
extract_cpu_time() {
    local file=$1
    grep "CPU time:" "$file" | tail -1 | awk '{print $NF}' | sed 's/s//'
}

echo "Experiment 1: THREADS_PER_BLOCK_1D Variations"
echo "----------------------------------------------"
printf "%-30s %-15s %-15s %-15s\n" "Configuration" "GPU Time (s)" "Accuracy (%)" "Speedup"
echo "----------------------------------------------"

for threads in 128 256 512; do
    file="$RESULTS_DIR/exp1_threads1d_${threads}.txt"
    if [ -f "$file" ]; then
        gpu_time=$(extract_gpu_time "$file")
        cpu_time=$(extract_cpu_time "$file")
        accuracy=$(extract_accuracy "$file")
        if [ -n "$gpu_time" ] && [ -n "$accuracy" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
            printf "%-30s %-15s %-15s %-15s\n" "THREADS=${threads}" "$gpu_time" "$accuracy" "${speedup}x"
        fi
    fi
done
echo ""

echo "Experiment 2: THREADS_PER_BLOCK_2D Variations"
echo "----------------------------------------------"
printf "%-30s %-15s %-15s %-15s\n" "Configuration" "GPU Time (s)" "Accuracy (%)" "Speedup"
echo "----------------------------------------------"

for blocksize in 8 16 32; do
    file="$RESULTS_DIR/exp2_threads2d_${blocksize}x${blocksize}.txt"
    if [ -f "$file" ]; then
        gpu_time=$(extract_gpu_time "$file")
        cpu_time=$(extract_cpu_time "$file")
        accuracy=$(extract_accuracy "$file")
        if [ -n "$gpu_time" ] && [ -n "$accuracy" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
            printf "%-30s %-15s %-15s %-15s\n" "BLOCK=${blocksize}x${blocksize}" "$gpu_time" "$accuracy" "${speedup}x"
        fi
    fi
done
echo ""

echo "Experiment 3: Hidden Layer Size Variations"
echo "----------------------------------------------"
printf "%-30s %-15s %-15s %-15s\n" "Configuration" "GPU Time (s)" "Accuracy (%)" "Speedup"
echo "----------------------------------------------"

for hidden in 128 256 512 1024; do
    file="$RESULTS_DIR/exp3_hidden_${hidden}.txt"
    if [ -f "$file" ]; then
        gpu_time=$(extract_gpu_time "$file")
        cpu_time=$(extract_cpu_time "$file")
        accuracy=$(extract_accuracy "$file")
        if [ -n "$gpu_time" ] && [ -n "$accuracy" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
            printf "%-30s %-15s %-15s %-15s\n" "HIDDEN=${hidden}" "$gpu_time" "$accuracy" "${speedup}x"
        fi
    fi
done
echo ""

echo "Experiment 4: Learning Rate Variations"
echo "----------------------------------------------"
printf "%-30s %-15s %-15s %-15s\n" "Configuration" "GPU Time (s)" "Accuracy (%)" "Speedup"
echo "----------------------------------------------"

for lr in 0.001 0.01 0.1; do
    file="$RESULTS_DIR/exp4_lr_${lr}.txt"
    if [ -f "$file" ]; then
        gpu_time=$(extract_gpu_time "$file")
        cpu_time=$(extract_cpu_time "$file")
        accuracy=$(extract_accuracy "$file")
        if [ -n "$gpu_time" ] && [ -n "$accuracy" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
            printf "%-30s %-15s %-15s %-15s\n" "LR=${lr}" "$gpu_time" "$accuracy" "${speedup}x"
        fi
    fi
done
echo ""

echo "Experiment 5: EVAL_BATCH_SIZE Variations"
echo "----------------------------------------------"
printf "%-30s %-15s %-15s %-15s\n" "Configuration" "GPU Time (s)" "Accuracy (%)" "Speedup"
echo "----------------------------------------------"

for batch in 64 256 512; do
    file="$RESULTS_DIR/exp5_evalbatch_${batch}.txt"
    if [ -f "$file" ]; then
        gpu_time=$(extract_gpu_time "$file")
        cpu_time=$(extract_cpu_time "$file")
        accuracy=$(extract_accuracy "$file")
        if [ -n "$gpu_time" ] && [ -n "$accuracy" ]; then
            speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
            printf "%-30s %-15s %-15s %-15s\n" "EVAL_BATCH=${batch}" "$gpu_time" "$accuracy" "${speedup}x"
        fi
    fi
done
echo ""

echo "=========================================="
echo "Analysis Summary"
echo "=========================================="

# Find best performing configuration overall
echo "Files in $RESULTS_DIR:"
ls -la "$RESULTS_DIR" 2>/dev/null | grep -E "exp[0-9]_" || echo "No experiment results found yet."

echo ""
echo "To integrate results into the report:"
echo "1. Review the results above"
echo "2. Update REPORT.tex with key findings"
echo "3. Highlight performance trends and GPU optimization insights"

#!/bin/bash

# Define the executable path
EXECUTABLE="./build/sample04-rohan csv /tmp/BarnesHutBenchmarks/synthetic100M.csv"

# Run the executable and pipe the output to a new file for each iteration
for ((i=1; i<=5; i++))
do
    OUTPUT_FILE="output_$i.txt"
    echo "Running iteration $i, output file: $OUTPUT_FILE"
    $EXECUTABLE > "$OUTPUT_FILE"
    echo "Iteration $i complete"
done

echo "All iterations complete"
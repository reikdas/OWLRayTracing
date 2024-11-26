#!/bin/bash

# Parameters
NUM_ITERATIONS=$2  # Number of iterations
if [ -z "$NUM_ITERATIONS" ]; then
    NUM_ITERATIONS=1
fi
experiment_type_argument=$1 # which experiment to run (changa, treelogy, scalability, sanitycheck) 


if [ "$experiment_type_argument" == "changa" ]; then
    echo "Running changa experiments..."
    FILES=("./synthetic25M.csv" "./dwf1_2048_00384.csv" "./dwf1_6144_01472.csv" "./lambb.csv")  # List of benchmarks for changa
    TYPE="csv"
elif [ "$experiment_type_argument" == "treelogy" ]; then
    echo "Running treelogy experiments..."
    TYPE="treelogy"
    FILES=("./treelogy_synthetic_10M.txt" "./treelogy_synthetic_25M.txt" "./treelogy_synthetic_50M.txt")  # List of benchmarks for treelogy
elif [ "$experiment_type_argument" == "scalability" ]; then
    echo "Running scalability experiments..."
    TYPE="csv"
    FILES=("./synthetic10M.csv" "./synthetic25M.csv" "./synthetic50M.csv" "./synthetic100M.csv")  # List of benchmarks for scalability
elif [ "$experiment_type_argument" == "sanitycheck" ]; then
    echo "Running sanity check..."
    TYPE="treelogy"
    FILES=("./treelogy_synthetic_1M.txt")  # List of benchmarks for scalability
else
    echo "Invalid argument. Choose changa, treelogy, or scalability argument."
    exit 1
fi

# Define the executable path
BUILD_DIR="./build"
EXECUTABLE="$BUILD_DIR/s01-rtbarneshut $TYPE"

# Build the executable
echo "Cleaning and building the project..."
if [ -d "$BUILD_DIR" ]; then
    (cd "$BUILD_DIR" && make clean && make rtbarneshut)
    if [ $? -ne 0 ]; then
        echo "Build failed. Exiting."
        exit 1
    fi
else
    echo "Build directory not found: $BUILD_DIR"
    exit 1
fi

# Function to extract values and calculate averages
calculate_averages() {
    local FILE="$1"
    local PREPROCESS_TOTAL=0
    local FORCE_CALC_TOTAL=0
    local ITERATIVE_STEP_TOTAL=0
    
    for ((i=1; i<=NUM_ITERATIONS; i++))
    do
        OUTPUT_FILE="./outputs/${BASENAME_NO_EXT}_output_$i.txt"
        
        # Extract the times from the output file
        PREPROCESS_TIME=$(grep -E "Preprocessing Time" "$OUTPUT_FILE" | awk '{print $3}')
        FORCE_CALC_TIME=$(grep -E "RT Cores Force Calculations time" "$OUTPUT_FILE" | awk '{print $6}')
        ITERATIVE_STEP_TIME=$(grep -E "Execution time" "$OUTPUT_FILE" | awk '{print $3}')
        
        # Accumulate totals
        PREPROCESS_TOTAL=$(echo "$PREPROCESS_TOTAL + $PREPROCESS_TIME" | bc)
        FORCE_CALC_TOTAL=$(echo "$FORCE_CALC_TOTAL + $FORCE_CALC_TIME" | bc)
        ITERATIVE_STEP_TOTAL=$(echo "$ITERATIVE_STEP_TOTAL + $ITERATIVE_STEP_TIME" | bc)
    done

    # Calculate averages
    PREPROCESS_AVG=$(echo "scale=6; $PREPROCESS_TOTAL / $NUM_ITERATIONS" | bc)
    FORCE_CALC_AVG=$(echo "scale=6; $FORCE_CALC_TOTAL / $NUM_ITERATIONS" | bc)
    ITERATIVE_STEP_AVG=$(echo "scale=6; $ITERATIVE_STEP_TOTAL / $NUM_ITERATIONS" | bc)

    # Print averages
    echo "Averages for $FILE:"
    echo "Preprocessing Time Average: $PREPROCESS_AVG seconds"
    echo "RT Cores Force Calculations Time Average: $FORCE_CALC_AVG seconds"
    echo "Execution Time Average: $ITERATIVE_STEP_AVG seconds"
}

# Run the executable for each file and iteration
for FILE in "${FILES[@]}"
do
    BASENAME=$(basename "$FILE")  # Extract the file name without the path
    BASENAME_NO_EXT="${BASENAME%.csv}"  # Remove the extension
    echo "Processing file: $FILE"
    
    for ((i=1; i<=NUM_ITERATIONS; i++))
    do
        OUTPUT_FILE="./outputs/${BASENAME_NO_EXT}_output_$i.txt"
        echo "Running iteration $i for file: $FILE, output file: $OUTPUT_FILE"
        $EXECUTABLE "$FILE" > "$OUTPUT_FILE"
        echo "Iteration $i complete for file: $FILE"
    done
    
    # Calculate averages after iterations
    calculate_averages "$FILE"
done
echo "All iterations complete for all files"
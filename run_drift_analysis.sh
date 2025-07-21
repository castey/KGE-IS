#!/bin/bash

# Default number of epochs
EPOCHS=${1:-5}  # Use first argument, or default to 5

# Ensure output folders exist
mkdir -p gen_data img

# Step 1: Generate drift data with specified number of epochs
echo "Generating drift data (${EPOCHS} epochs)..."
DATAFILE="gen_data/drift_data_${EPOCHS}epochs.json"
python3 save_drift_data.py --epochs "$EPOCHS" --filename "$DATAFILE"

# Step 2: Define SPO elements
spo_elements=("head" "relation" "tail")

# Step 3: Loop through SPO terms
for spo in "${spo_elements[@]}"; do
    base_name=$(basename "$DATAFILE" .json)
    output_file="img/${base_name}_${spo}.png"
    
    echo "Generating drift histogram for $spo from $DATAFILE..."
    python3 calc_drift.py --filepath "$DATAFILE" --spo "$spo" --filename "$output_file"
done

echo "✅ Drift analysis completed for ${EPOCHS} epochs."

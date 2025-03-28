import torch
import os
from itertools import combinations

# This script compares the matrices in the initial and trained folders and shows whichs ones are the same and which ones are different...
# The results might make sense, if not, oh well

# Define paths
INITIAL_MATRICES_DIR = "./matrices_initial"
TRAINED_MATRICES_DIR = "./matrices_trained"

# Function to load all matrices from a directory
def load_matrices(directory):
    """Loads all .pt matrices from a given directory and returns a dictionary."""
    matrices = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pt"):
            filepath = os.path.join(directory, filename)
            matrices[filename] = torch.load(filepath)
    return matrices

# Load initial and trained matrices
initial_matrices = load_matrices(INITIAL_MATRICES_DIR)
trained_matrices = load_matrices(TRAINED_MATRICES_DIR)

# Combine all matrices into a single dictionary
all_matrices = {**initial_matrices, **trained_matrices}

# Compare matrices pairwise
print("\n=== Matrix Comparison Results ===")
comparison_results = []

for (name1, mat1), (name2, mat2) in combinations(all_matrices.items(), 2):
    are_identical = torch.equal(mat1, mat2)
    comparison_results.append((name1, name2, are_identical))

    if are_identical:
        print(f"[MATCH] {name1} == {name2}")
    else:
        print(f"[DIFFERENT] {name1} != {name2}")

# Summary
num_matches = sum(1 for _, _, match in comparison_results if match)
num_differences = len(comparison_results) - num_matches

print("\n=== Summary ===")
print(f"Total Matches: {num_matches}")
print(f"Total Differences: {num_differences}")

print("\nComparison complete!")

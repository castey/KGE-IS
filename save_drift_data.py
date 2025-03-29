from generate_drift_data import generate_drift_data
import json

def save_drift_data_to_file(filename="drift_data.json"):
    """
    Calls generate_drift_data() to obtain embedding drift information and saves it to a JSON file.
    
    Args:
    - filename (str): The name of the file to save the drift data.
    """
    print("Generating drift data...")
    drift_data = generate_drift_data()
    
    # Convert NumPy arrays to lists for JSON serialization
    serializable_data = {
        triple: {
            dataset: {key: value.tolist() for key, value in embeddings.items()}
            for dataset, embeddings in dataset_data.items()
        }
        for triple, dataset_data in drift_data.items()
    }
    
    # Save as JSON file
    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Drift data saved to {filename}")

save_drift_data_to_file()

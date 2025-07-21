from generate_drift_data import generate_drift_data
import pandas as pd
import json
import argparse
import os

def save_drift_data(format_choice="json", filename=None, num_epochs=100):
    print(f"Generating drift data (epochs={num_epochs})...")
    drift_data = generate_drift_data(num_epochs=num_epochs)
    print("Drift data obtained.")

    if format_choice == "csv":
        if filename is None:
            filename = "drift_data.csv"
        # ensure directory exists if path includes one
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        print("Preparing data for CSV format...")

        rows = []
        for triple, dataset_embeddings in drift_data.items():
            for dataset, components in dataset_embeddings.items():
                row = {
                    "triple": triple,
                    "dataset": dataset
                }
                for component_name, embedding in components.items():
                    for i, val in enumerate(embedding):
                        row[f"{component_name}_{i}"] = val
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        print(f"Drift data saved as CSV to {filename}")

    else:  # JSON
        if filename is None:
            filename = "drift_data.json"
        # ensure directory exists if path includes one
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        print("Preparing data for JSON format...")

        serializable_data = {
            triple: {
                dataset: {key: value.tolist() for key, value in embeddings.items()}
                for dataset, embeddings in dataset_data.items()
            }
            for triple, dataset_data in drift_data.items()
        }

        with open(filename, "w") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Drift data saved as JSON to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save drift data to CSV or JSON.")
    parser.add_argument("--format", choices=["csv", "json"], default="json", help="Output format (json or csv).")
    parser.add_argument("--filename", default=None, help="Output filename (default: drift_data.json or drift_data.csv).")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to pass to generate_drift_data()")

    args = parser.parse_args()
    save_drift_data(format_choice=args.format, filename=args.filename, num_epochs=args.epochs)

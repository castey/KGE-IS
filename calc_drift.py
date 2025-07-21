import ijson
from scipy.spatial.distance import euclidean
import statistics
import matplotlib.pyplot as plt
import os
import argparse

def calc_drft(file_path, spo_term, output_filename=None):
    if output_filename is None:
        output_filename = f"drift_histogram_{spo_term}.png"

    # Ensure directory exists
    dir_path = os.path.dirname(output_filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    print(f"Saving plot to: {output_filename}")

    # Normalize RDF-style terms
    spo_term = {'subject': 'head', 'predicate': 'relation', 'object': 'tail'}.get(spo_term, spo_term)

    distances_1, distances_2, distances_3 = [], [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        triples = ijson.kvitems(f, '')
        for triple_key, triple_data in triples:
            try:
                h237 = triple_data['dataset_237'][spo_term]
                h238 = triple_data['dataset_238'][spo_term]
                h239 = triple_data['dataset_239'][spo_term]

                distances_1.append(euclidean(h237, h238))
                distances_2.append(euclidean(h237, h239))
                distances_3.append(euclidean(h238, h239))
            except KeyError as e:
                print(f"Skipping {triple_key} due to missing key: {e}")

    def summarize(distances):
        return {
            'mean': statistics.mean(distances),
            'std_dev': statistics.stdev(distances)
        }

    summary_1 = summarize(distances_1)
    summary_2 = summarize(distances_2)
    summary_3 = summarize(distances_3)

    # Output to console
    print("Δ₁ (237 vs 238):")
    print(f"  Mean Euclidean Drift: {summary_1['mean']:.6f}")
    print(f"  Std Dev:              {summary_1['std_dev']:.6f}\n")

    print("Δ₂ (237 vs 239):")
    print(f"  Mean Euclidean Drift: {summary_2['mean']:.6f}")
    print(f"  Std Dev:              {summary_2['std_dev']:.6f}\n")

    print("Δ₃ (238 vs 239):")
    print(f"  Mean Euclidean Drift: {summary_3['mean']:.6f}")
    print(f"  Std Dev:              {summary_3['std_dev']:.6f}")

    # Save histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances_1, bins=500, alpha=0.6, label='Δ₁ (237 vs 238)')
    plt.hist(distances_2, bins=500, alpha=0.6, label='Δ₂ (237 vs 239)')
    plt.hist(distances_3, bins=500, alpha=0.6, label='Δ₃ (238 vs 239)')
    plt.title(f'Euclidean Drift Distribution ({spo_term}s)')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)

    # Generate markdown content
    md_output_path = os.path.splitext(output_filename)[0] + ".md"
    image_filename = os.path.basename(output_filename)

    md_content = f"""### Drift Summary for `{spo_term}`

| Comparison         | Mean Euclidean Drift | Standard Deviation |
|--------------------|----------------------|---------------------|
| **Δ₁ (237 vs 238)** | {summary_1['mean']:.6f}             | {summary_1['std_dev']:.6f}           |
| **Δ₂ (237 vs 239)** | {summary_2['mean']:.6f}             | {summary_2['std_dev']:.6f}           |
| **Δ₃ (238 vs 239)** | {summary_3['mean']:.6f}             | {summary_3['std_dev']:.6f}           |

![{spo_term} drift histogram]({image_filename})
"""

    with open(md_output_path, 'w', encoding='utf-8') as md_file:
        md_file.write(md_content)

    print(f"Markdown summary saved to: {md_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot drift histogram.")
    parser.add_argument("--filepath", default="./drift_data.json", help="Path to the drift data JSON file.")
    parser.add_argument("--spo", default="head", choices=["head", "relation", "tail"], help="Which element to calculate drift for (head, relation, tail).")
    parser.add_argument("--filename", default=None, help="Optional output filename (default: drift_histogram_<spo>.png)")
    args = parser.parse_args()

    calc_drft(file_path=args.filepath, spo_term=args.spo, output_filename=args.filename)

import argparse
import os
import matplotlib.pyplot as plt
import ijson
from scipy.spatial.distance import euclidean

def calc_top10_drift(file_path, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    def plot_bar(data, title_suffix, filename_suffix, spo_term):
        labels = [label for label, _ in data]
        values = [value for _, value in data]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, values)

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylabel('Drift Amount (Euclidean Distance)')
        plt.title(f"Top 10 {title_suffix} ({spo_term.capitalize()}s)")
        plt.tight_layout()

        # Add drift value above each bar
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{value:.4f}", 
                     ha='center', va='bottom', fontsize=7, rotation=90)

        save_path = os.path.join(output_dir, f"{spo_term}_{filename_suffix}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot: {save_path}")

    def plot_line(data, filename_suffix, spo_term):
        sorted_data = sorted(data, key=lambda x: x[1])
        labels = [label for label, _ in sorted_data]
        values = [value for _, value in sorted_data]

        plt.figure(figsize=(14, 6))
        plt.plot(range(len(values)), values, marker='o', markersize=2, linewidth=1)
        plt.xlabel('Entity Index (sorted by drift)')
        plt.ylabel('Drift Amount (Euclidean Distance)')
        plt.title(f"Drift Distribution: {filename_suffix} ({spo_term.capitalize()}s)")
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{spo_term}_{filename_suffix}_line.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved line plot: {save_path}")
    spo_term = "tail"
    # Now start processing normally
    print(f"Processing: {spo_term}s")

    drift_1 = []  # 237 vs 238
    drift_2 = []  # 237 vs 239
    drift_3 = []  # 238 vs 239

    with open(file_path, 'r', encoding='utf-8') as f:
        triples = ijson.kvitems(f, '')
        for triple_key, triple_data in triples:
            try:
                h237 = triple_data['dataset_237'][spo_term]
                h238 = triple_data['dataset_238'][spo_term]
                h239 = triple_data['dataset_239'][spo_term]

                parts = triple_key.split('\t')
                if spo_term == 'head':
                    label = parts[0]
                elif spo_term == 'tail':
                    label = parts[2]
                else:
                    print(f"triple_key {triple_key} parsing failure!")
                    continue

                drift_1.append((label, euclidean(h237, h238)))
                drift_2.append((label, euclidean(h237, h239)))
                drift_3.append((label, euclidean(h238, h239)))
                
            except (KeyError, IndexError) as e:
                print(f"Skipping {triple_key} due to missing key or bad format: {e}")

    for drift_set, name in [(drift_1, "237_vs_238"), (drift_2, "237_vs_239"), (drift_3, "238_vs_239")]:
        if not drift_set:
            continue

        unique_drift = {}
        for label, value in drift_set:
            if label not in unique_drift or value > unique_drift[label]:
                unique_drift[label] = value
        drift_set = list(unique_drift.items())

        highest_10 = sorted(drift_set, key=lambda x: x[1], reverse=True)[:10]

        # Actually plot now
        plot_bar(highest_10, f"Highest Drift {name}", f"{name}_highest_{spo_term}", spo_term)
        plot_line(drift_set, name, spo_term)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot top 10 drifts for heads and tails.")
    parser.add_argument("--filepath", default="./drift_data.json", help="Path to the drift data JSON file.")
    parser.add_argument("--output", default="./output", help="Directory to save output plots.")
    args = parser.parse_args()

    calc_top10_drift(file_path=args.filepath, output_dir=args.output)

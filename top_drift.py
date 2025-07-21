import argparse
import os
import matplotlib.pyplot as plt
import ijson
from scipy.spatial.distance import euclidean
import requests

import time
import random

wikidata_cache = {}

def with_exponential_backoff(request_fn, max_retries=5, base_delay=1.0):
    """Retries a request function using exponential backoff."""
    for attempt in range(max_retries):
        try:
            return request_fn()
        except requests.exceptions.RequestException as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    raise Exception(f"All {max_retries} attempts failed.")

def get_wikidata_label(entity_id):
    """Retrieves the English label for a Wikidata entity ID, handling Freebase IDs."""
    if entity_id in wikidata_cache:
        return wikidata_cache[entity_id]

    headers = {
        'Accept': 'application/sparql-results+json',
        'User-Agent': 'Knowledge Graph Embedding Drift Study (castro.31@example.com)'
    }


    if entity_id.startswith('/m/'):  # Freebase ID
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item wdt:P646 "{entity_id}" .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """
        url = "https://query.wikidata.org/sparql"
        def request_fn():
            response = requests.get(url, params={'query': query}, headers=headers, timeout=10)
            response.raise_for_status()

            if not response.content:
                raise ValueError("Empty response body")

            content_type = response.headers.get("Content-Type", "")
            if "json" not in content_type.lower():
                raise ValueError(f"Unexpected content type: {content_type}")

            results = response.json().get('results', {}).get('bindings', [])

            if results:
                label = results[0]['itemLabel']['value']
                wikidata_cache[entity_id] = label
                return label
            else:
                wikidata_cache[entity_id] = entity_id
                return entity_id

        return with_exponential_backoff(request_fn)

    elif entity_id.startswith('Q') and entity_id[1:].isdigit():  # Wikidata ID
        url = f"https://www.wikidata.org/w/api.php"
        def request_fn():
            response = requests.get(url, params={
                "action": "wbgetentities",
                "ids": entity_id,
                "props": "labels",
                "languages": "en",
                "format": "json"
            }, headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            label = data.get('entities', {}).get(entity_id, {}).get('labels', {}).get('en', {}).get('value')
            wikidata_cache[entity_id] = label if label else entity_id
            return wikidata_cache[entity_id]

        return with_exponential_backoff(request_fn)

    else:
        wikidata_cache[entity_id] = entity_id
        return entity_id


def calc_top_bottom_10_drift_wikidata(file_path, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)

    def plot_bar(data, title_suffix, filename_suffix, spo_term, top=True):
        if not data:
            print(f"No data to plot for {title_suffix} ({spo_term.capitalize()}s).")
            return

        labels_with_wikidata = []
        for label, value in data:
            wikidata_label = get_wikidata_label(label)
            labels_with_wikidata.append(wikidata_label)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels_with_wikidata, [value for _, value in data], color='skyblue' if top else 'salmon')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.ylabel('Drift Amount (Euclidean Distance)')
        title_prefix = "Top" if top else "Least"
        plt.title(f"{title_prefix} 10 {title_suffix} ({spo_term.capitalize()}s)")
        plt.tight_layout()

        # Add drift value above each bar
        for bar, value in zip(bars, [value for _, value in data]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{value:.4f}",
                     ha='center', va='bottom', fontsize=7, rotation=90)

        filename_prefix = "top" if top else "bottom"
        save_path = os.path.join(output_dir, f"{spo_term}_wikidata_{filename_suffix}_{filename_prefix}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot: {save_path}")

    for spo_term in ["head"]:
        print(f"Processing: {spo_term}s with Wikidata labels")

        drift_1 = []  # 237 vs 238
        drift_2 = []  # 237 vs 239
        drift_3 = []  # 238 vs 239

        with open(file_path, 'r', encoding='utf-8') as f:
            triples = ijson.kvitems(f, '')
            for triple_key, triple_data in triples:
                try:
                    e237 = triple_data['dataset_237'][spo_term]
                    e238 = triple_data['dataset_238'][spo_term]
                    e239 = triple_data['dataset_239'][spo_term]

                    parts = triple_key.strip().split('\t')
                    if spo_term == 'head':
                        label = parts[0]
                    elif spo_term == 'tail':
                        label = parts[2]
                    else:
                        continue

                    drift_1.append((label, euclidean(e237, e238)))
                    drift_2.append((label, euclidean(e237, e239)))
                    drift_3.append((label, euclidean(e238, e239)))

                except (KeyError, IndexError, ValueError) as e:
                    print(f"Skipping {triple_key}: {e}")
                    continue

        for drift_set, name in [(drift_1, "237_vs_238"), (drift_2, "237_vs_239"), (drift_3, "238_vs_239")]:
            if not drift_set:
                continue

            unique_drift = {}
            for label, value in drift_set:
                if label not in unique_drift or value > unique_drift[label]:
                    unique_drift[label] = value

            drift_list = list(unique_drift.items())
            sorted_drift = sorted(drift_list, key=lambda x: x[1])

            top_10 = sorted_drift[-10:][::-1]
            bottom_10 = sorted_drift[:10]

            plot_bar(top_10, f"Highest Drift {name}", f"{name}_highest", spo_term, top=True)
            plot_bar(bottom_10, f"Lowest Drift {name}", f"{name}_lowest", spo_term, top=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and plot top/bottom 10 drifts for heads and tails with Wikidata labels (including Freebase IDs).")
    parser.add_argument("-f", "--filepath", default="./drift_data.json", help="Path to the drift data JSON file.")
    parser.add_argument("-o", "--output", default="./output", help="Directory to save output plots.")

    args = parser.parse_args()

    calc_top_bottom_10_drift_wikidata(file_path=args.filepath, output_dir=args.output)

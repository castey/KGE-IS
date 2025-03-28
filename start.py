import torch
import random
from pykeen.pipeline import pipeline
from pykeen.datasets import Kinships
from colorama import Fore, Style

# Function to set randomness behavior
def set_randomness(deterministic=True, seed=42):
    """Sets randomness for reproducibility or allows full randomness.

    Args:
        deterministic (bool): If True, sets fixed seeds for reproducibility.
        seed (int): The seed value to use when deterministic mode is enabled.
    """
    if deterministic:
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        random.seed()  # Uses system entropy for Python random
        torch.seed()   # Uses system entropy for PyTorch random

# Toggle this flag to control randomness behavior
DETERMINISTIC_MODE = True  # Set to True for fixed selection, False for randomness

# Apply the randomness setting
set_randomness(deterministic=DETERMINISTIC_MODE)

# Load the dataset
print(Fore.YELLOW + "Loading the Nations dataset..." + Style.RESET_ALL)
dataset = Kinships()

# Train a TransE model with deterministic behavior
print(Fore.YELLOW + "Training the TransE model on Nations dataset..." + Style.RESET_ALL)
set_randomness(deterministic=True)  # Ensure model training is deterministic
result = pipeline(
    training=dataset.training,
    testing=dataset.testing,
    validation=dataset.validation,
    model="TransE",
    random_seed=42,
)

# Extract trained model and triples
model = result.model
triples_factory = result.training
all_triples = triples_factory.mapped_triples

print(Fore.GREEN + "Model training complete!\n" + Style.RESET_ALL)

# Restore randomness for data selection
set_randomness(deterministic=DETERMINISTIC_MODE)

# Function to get embeddings for triples
def get_embeddings(triples, model):
    head_emb = model.entity_representations[0](triples[:, 0])
    relation_emb = model.relation_representations[0](triples[:, 1])
    tail_emb = model.entity_representations[0](triples[:, 2])
    return head_emb + relation_emb - tail_emb  # TransE: h + r - t

# Distance metrics
def cosine_distance(tensor_a, tensor_b):
    return 1 - torch.nn.functional.cosine_similarity(tensor_a.mean(dim=0, keepdim=True), tensor_b.mean(dim=0, keepdim=True))

def euclidean_distance(tensor_a, tensor_b):
    return torch.norm(tensor_a.mean(dim=0) - tensor_b.mean(dim=0), p=2)

# Choose a single trial to illustrate
print(Fore.CYAN + "=== Example Trial: Original and Altered Data ===" + Style.RESET_ALL)

# Sample 5 random triples
sampled_indices = torch.randperm(len(all_triples))[:5]  # Random selection
V1 = all_triples[sampled_indices]

# Remove one random triple to create V2
removed_index = torch.randint(0, 5, (1,)).item()  # Select a random index to remove
V2 = torch.cat([V1[:removed_index], V1[removed_index + 1:]])
V_delta = V1[removed_index].unsqueeze(0)  # Single removed triple

# Convert to human-readable format
def convert_to_readable(triples):
    return [
        (
            triples_factory.entity_id_to_label[triple[0].item()],
            triples_factory.relation_id_to_label[triple[1].item()],
            triples_factory.entity_id_to_label[triple[2].item()]
        )
        for triple in triples
    ]

V1_readable = convert_to_readable(V1)
V2_readable = convert_to_readable(V2)
V_delta_readable = convert_to_readable(V_delta)

# Display before alteration
print(Fore.GREEN + "\nOriginal Sampled Set (V1 - 5 triples):" + Style.RESET_ALL)
for triple in V1_readable:
    print(f"  {triple[0]} --[{triple[1]}]--> {triple[2]}")

print(Fore.RED + "\nRemoved Triple (VΔ):" + Style.RESET_ALL)
print(f"  {V_delta_readable[0][0]} --[{V_delta_readable[0][1]}]--> {V_delta_readable[0][2]}")

print(Fore.YELLOW + "\nRemaining Set After Removal (V2 - 4 triples):" + Style.RESET_ALL)
for triple in V2_readable:
    print(f"  {triple[0]} --[{triple[1]}]--> {triple[2]}")

# Get embeddings
V1_emb = get_embeddings(V1, model)
V2_emb = get_embeddings(V2, model)
V_delta_emb = get_embeddings(V_delta, model)

# Compute distances
cos_v1_v2 = cosine_distance(V1_emb, V2_emb).item()
cos_v1_vd = cosine_distance(V1_emb, V_delta_emb).item()
cos_v2_vd = cosine_distance(V2_emb, V_delta_emb).item()

euc_v1_v2 = euclidean_distance(V1_emb, V2_emb).item()
euc_v1_vd = euclidean_distance(V1_emb, V_delta_emb).item()
euc_v2_vd = euclidean_distance(V2_emb, V_delta_emb).item()

# Print results for this trial
print(Fore.BLUE + "\n=== Embedding Distances for Example Trial ===" + Style.RESET_ALL)
print(f"Cosine Distances:")
print(f"  V1 vs V2: {cos_v1_v2:.4f}")
print(f"  V1 vs VΔ: {cos_v1_vd:.4f}")
print(f"  V2 vs VΔ: {cos_v2_vd:.4f}")

print(f"\nEuclidean Distances:")
print(f"  V1 vs V2: {euc_v1_v2:.4f}")
print(f"  V1 vs VΔ: {euc_v1_vd:.4f}")
print(f"  V2 vs VΔ: {euc_v2_vd:.4f}")

# Perform multiple trials
num_samples = 1000
cosine_distances = {"V1 vs V2": [], "V1 vs VΔ": [], "V2 vs VΔ": []}
euclidean_distances = {"V1 vs V2": [], "V1 vs VΔ": [], "V2 vs VΔ": []}

for _ in range(num_samples):
    sampled_indices = torch.randperm(len(all_triples))[:5]  # Random selection
    V1 = all_triples[sampled_indices]

    removed_index = torch.randint(0, 5, (1,)).item()
    V2 = torch.cat([V1[:removed_index], V1[removed_index + 1:]])
    V_delta = V1[removed_index].unsqueeze(0)

    V1_emb = get_embeddings(V1, model)
    V2_emb = get_embeddings(V2, model)
    V_delta_emb = get_embeddings(V_delta, model)

    cosine_distances["V1 vs V2"].append(cosine_distance(V1_emb, V2_emb).item())
    cosine_distances["V1 vs VΔ"].append(cosine_distance(V1_emb, V_delta_emb).item())
    cosine_distances["V2 vs VΔ"].append(cosine_distance(V2_emb, V_delta_emb).item())

    euclidean_distances["V1 vs V2"].append(euclidean_distance(V1_emb, V2_emb).item())
    euclidean_distances["V1 vs VΔ"].append(euclidean_distance(V1_emb, V_delta_emb).item())
    euclidean_distances["V2 vs VΔ"].append(euclidean_distance(V2_emb, V_delta_emb).item())

# Compute averages
average_cosine = {key: sum(values) / num_samples for key, values in cosine_distances.items()}
average_euclidean = {key: sum(values) / num_samples for key, values in euclidean_distances.items()}

print(Fore.MAGENTA + "\n=== Final Average Results from " + str(num_samples) + " Trials ===" + Style.RESET_ALL)

print(Fore.BLUE + "\nAverage Cosine Distances:" + Style.RESET_ALL)
for key, value in average_cosine.items():
    print(f"{key}: {value:.4f}")

print(Fore.BLUE + "\nAverage Euclidean Distances:" + Style.RESET_ALL)
for key, value in average_euclidean.items():
    print(f"{key}: {value:.4f}")

print(Fore.GREEN + "\nExperiment complete!" + Style.RESET_ALL)

## 
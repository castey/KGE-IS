import torch
import random
import os
import pandas as pd
import datetime
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from colorama import Fore, Style

# create readme about makng pykeen deterministic
# look into how pykeen takes hyper-parameters (batch size, loss function, etc)
# task: train pykeen transE on 237, train on 238 using the same start point weighted matrix -> compare distances between like vectors in 237 and 238 

# Function to set randomness behavior
def set_randomness(deterministic=True, seed=42):
    if deterministic:
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        random.seed()
        torch.manual_seed(torch.seed())

DETERMINISTIC_MODE = True  # Toggle for reproducibility
set_randomness(deterministic=DETERMINISTIC_MODE)

# Dataset paths
DATASET_PATH = "./ds/fb15k-237"
TRAIN_FILE = os.path.join(DATASET_PATH, "train.txt")
VALID_FILE = os.path.join(DATASET_PATH, "valid.txt")
TEST_FILE = os.path.join(DATASET_PATH, "test.txt")
ENTITIES_FILE = os.path.join(DATASET_PATH, "entities.tsv")
RELATIONS_FILE = os.path.join(DATASET_PATH, "relations.tsv")

# Matrix storage paths
INITIAL_MATRICES_DIR = "./matrices_initial"
TRAINED_MATRICES_DIR = "./matrices_trained"
os.makedirs(INITIAL_MATRICES_DIR, exist_ok=True)
os.makedirs(TRAINED_MATRICES_DIR, exist_ok=True)

# Function to generate filenames based on sequence
def get_next_filename(directory, prefix):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    existing_files = [f for f in os.listdir(directory) if f.startswith(f"{prefix}_{today}_")]

    sequence_numbers = []
    for f in existing_files:
        try:
            num = int(f.split("_")[-1].split(".")[0])
            sequence_numbers.append(num)
        except ValueError:
            continue

    next_number = max(sequence_numbers) + 1 if sequence_numbers else 1
    return os.path.join(directory, f"{prefix}_{today}_{next_number}.pt")

# Load dataset
print(Fore.YELLOW + "Loading the FB15k-237 dataset..." + Style.RESET_ALL)
training_factory = TriplesFactory.from_path(TRAIN_FILE)
validation_factory = TriplesFactory.from_path(VALID_FILE)
testing_factory = TriplesFactory.from_path(TEST_FILE)

# Load entity and relation mappings
entity_id_to_label = pd.read_csv(ENTITIES_FILE, sep="\t", header=None, index_col=0, names=["ID", "Label"])["Label"].to_dict()
relation_id_to_label = pd.read_csv(RELATIONS_FILE, sep="\t", header=None, index_col=0, names=["ID", "Label"])["Label"].to_dict()


# Initialize model without training to capture initial weights
print(Fore.YELLOW + "Initializing TransE model..." + Style.RESET_ALL)
model = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    model="TransE",
    training_kwargs=dict(num_epochs=0),  # No training to capture initial weights
    random_seed=42,
).model

# Save the initial weight matrix
initial_matrix_path = get_next_filename(INITIAL_MATRICES_DIR, "m_initial")
torch.save(model.entity_representations[0]._embeddings.weight.data.clone(), initial_matrix_path)
print(Fore.GREEN + f"Initial weight matrix saved to {initial_matrix_path}" + Style.RESET_ALL)


# Train the model
print(Fore.YELLOW + "Training the TransE model on FB15k-237 dataset..." + Style.RESET_ALL)
set_randomness(deterministic=True)
result = pipeline(
    training=training_factory,
    testing=testing_factory,
    validation=validation_factory,
    model="TransE",
    random_seed=42,
)

print(result)
model = result.model  # Get the trained model

# Save the trained weight matrix
trained_matrix_path = get_next_filename(TRAINED_MATRICES_DIR, "m_trained")
torch.save(model.entity_representations[0]._embeddings.weight.data.clone(), trained_matrix_path)
print(Fore.GREEN + f"Trained weight matrix saved to {trained_matrix_path}" + Style.RESET_ALL)

# Retrieve triples
all_triples = training_factory.mapped_triples

# Restore randomness
set_randomness(deterministic=DETERMINISTIC_MODE)

# Function to get embeddings
def get_embeddings(triples, model):
    head_emb = model.entity_representations[0](triples[:, 0])
    relation_emb = model.relation_representations[0](triples[:, 1])
    tail_emb = model.entity_representations[0](triples[:, 2])
    return head_emb + relation_emb - tail_emb  # TransE: h + r - t

# Similarity and distance metrics
def cosine_similarity(tensor_a, tensor_b):
    return torch.nn.functional.cosine_similarity(tensor_a.mean(dim=0, keepdim=True), tensor_b.mean(dim=0, keepdim=True))

def euclidean_distance(tensor_a, tensor_b):
    return torch.norm(tensor_a.mean(dim=0) - tensor_b.mean(dim=0), p=2)

# Track cumulative similarities and distances
total_cos_v1_v2 = 0.0
total_cos_v1_vd = 0.0
total_cos_v2_vd = 0.0
total_euc_v1_v2 = 0.0
total_euc_v1_vd = 0.0
total_euc_v2_vd = 0.0
num_iterations = 0

# Iterate through all triples in chunks of 5
for _ in range(len(all_triples) // 5):
    sampled_indices = torch.randperm(len(all_triples))[:5]
    V1 = all_triples[sampled_indices]

    removed_index = torch.randint(0, 5, (1,)).item()
    V2 = torch.cat([V1[:removed_index], V1[removed_index + 1:]])
    V_delta = V1[removed_index].unsqueeze(0)

    # Get embeddings
    V1_emb = get_embeddings(V1, model)
    V2_emb = get_embeddings(V2, model)
    V_delta_emb = get_embeddings(V_delta, model)

    # Compute similarities and distances
    cos_v1_v2 = cosine_similarity(V1_emb, V2_emb).item()
    cos_v1_vd = cosine_similarity(V1_emb, V_delta_emb).item()
    cos_v2_vd = cosine_similarity(V2_emb, V_delta_emb).item()

    euc_v1_v2 = euclidean_distance(V1_emb, V2_emb).item()
    euc_v1_vd = euclidean_distance(V1_emb, V_delta_emb).item()
    euc_v2_vd = euclidean_distance(V2_emb, V_delta_emb).item()

    # Accumulate results
    total_cos_v1_v2 += cos_v1_v2
    total_cos_v1_vd += cos_v1_vd
    total_cos_v2_vd += cos_v2_vd
    total_euc_v1_v2 += euc_v1_v2
    total_euc_v1_vd += euc_v1_vd
    total_euc_v2_vd += euc_v2_vd
    num_iterations += 1

# Compute averages
avg_cos_v1_v2 = total_cos_v1_v2 / num_iterations
avg_cos_v1_vd = total_cos_v1_vd / num_iterations
avg_cos_v2_vd = total_cos_v2_vd / num_iterations
avg_euc_v1_v2 = total_euc_v1_v2 / num_iterations
avg_euc_v1_vd = total_euc_v1_vd / num_iterations
avg_euc_v2_vd = total_euc_v2_vd / num_iterations

# Print results
print(Fore.BLUE + "\n=== Average Embedding Similarities & Distances ===" + Style.RESET_ALL)
print(f"Average Cosine Similarities:\n  V1 vs V2: {avg_cos_v1_v2:.4f}\n  V1 vs VΔ: {avg_cos_v1_vd:.4f}\n  V2 vs VΔ: {avg_cos_v2_vd:.4f}")
print(f"\nAverage Euclidean Distances:\n  V1 vs V2: {avg_euc_v1_v2:.4f}\n  V1 vs VΔ: {avg_euc_v1_vd:.4f}\n  V2 vs VΔ: {avg_euc_v2_vd:.4f}")

print(Fore.GREEN + "\nFull dataset experiment complete!" + Style.RESET_ALL)

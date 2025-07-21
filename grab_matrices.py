import torch
import random
import os
import pandas as pd
import datetime
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from colorama import Fore, Style

# this file just outputs the initial weighted matrix and the final weighted matrix after the training step
# demonstrates deterministic behavior when seeding random.seed and torch.manual_seed with static values 
# e.g. a seed of 42 will always produce the same final weighted matrix given the same training data

# Function to set randomness behavior
def set_randomness(deterministic=True, seed=42):
    if deterministic:
        random.seed(seed)
        torch.manual_seed(seed)
    else:
        random.seed()
        torch.manual_seed(torch.seed())

DETERMINISTIC_MODE = True  # false for random
set_randomness(deterministic=DETERMINISTIC_MODE)

# Dataset paths
DATASET_PATH = "../dataset/fb15k-237"
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

# Train the model this time allowing for complete training
print(Fore.YELLOW + "Training the TransE model on FB15k-237 dataset..." + Style.RESET_ALL)

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
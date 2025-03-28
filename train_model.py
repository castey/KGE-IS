import torch
import random
import os
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from colorama import Fore, Style



random.seed(42)
torch.manual_seed(42)

# Dataset paths
DATASET_PATH = "./ds/fb15k-237"
TRAIN_FILE = os.path.join(DATASET_PATH, "train.txt")
VALID_FILE = os.path.join(DATASET_PATH, "valid.txt")
TEST_FILE = os.path.join(DATASET_PATH, "test.txt")

# Output directory for saved models
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
print(Fore.YELLOW + "Loading the FB15k-237 dataset..." + Style.RESET_ALL)
training_factory = TriplesFactory.from_path(TRAIN_FILE)
validation_factory = TriplesFactory.from_path(VALID_FILE)
testing_factory = TriplesFactory.from_path(TEST_FILE)

# Train the model
print(Fore.YELLOW + "Training the TransE model on FB15k-237 dataset..." + Style.RESET_ALL)
result = pipeline(
    training=training_factory,
    validation=validation_factory,
    testing=testing_factory,
    model="TransE",
    random_seed=42,
)

# Save the full model and training results
timestamped_path = os.path.join(MODEL_DIR, f"transe_model_{result.start_datetime.date()}_{result.start_datetime.time().strftime('%H-%M-%S')}")
result.save_to_directory(timestamped_path)

print(Fore.GREEN + f"\nTrained model saved to: {timestamped_path}" + Style.RESET_ALL)

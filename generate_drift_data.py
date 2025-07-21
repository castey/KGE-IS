import torch
import random
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import os

# to set pykeen to deterministic simply set random.seed() and torch.manual_seed() to any number. 
# this seeds all random number generation with a fixed seed ensuring deterministic output

# to verify output is determinsitic pass in training_kwargs=dict(num_epochs=0) to pipeline()
# and grab the weighted matrix from that as well as an execution of pipeline() that is allowed to train fully (see grab_matrices.py)
# and compare across runs trained to trained and untrained to untrained (see compare_matrices.py)

# this makes pykeen deterministic
seed = 42
random.seed(seed)
torch.manual_seed(seed)

# function to train on each dataset and return an array of results
# this might be a little bit clunky but it will work on the 237,8,9 datasets specifically
def train_on_23x(num_epochs=100):
    
    # going to loop through the datasets and call pipeline() to train on each one
    # then push those results into an array
    # this might make comparing the embeddings simpler too
    result_array = []
    
    for i in range(3):
        x = i + 7
        
        # this should be the right path... 
        DS_PATH = "../dataset/fb15k-23" + str(x) 
        TRAIN = os.path.join(DS_PATH, "train.txt")
        VALID = os.path.join(DS_PATH, "valid.txt")
        TEST = os.path.join(DS_PATH, "test.txt")
        
        # load dataset
        training_factory = TriplesFactory.from_path(TRAIN)
        validation_factory = TriplesFactory.from_path(VALID)
        testing_factory = TriplesFactory.from_path(TEST)

        # call the pipeline to generate results from dataset 23x
        result = pipeline(
            training=training_factory,
            testing=testing_factory,
            validation=validation_factory,
            model="TransE",
            random_seed=seed,
            training_kwargs=dict(num_epochs=num_epochs),
        )
        
        result_array.append(result)

        
    return result_array

# this takes in the results of the training step and outputs the triples that are shared by all datasets
# that way we know we are comparing the embeddings of the same triples to each other
def sort_results_into_correspondence(result_array):
    # extract triples from each result (we'll use training triples here)
    triple_sets = []
    triple_lists = []
    
    for result in result_array:
        triples = result.training.mapped_triples  # Tensor of shape (n, 3)
        factory = result.training
        id_to_label = factory.entity_id_to_label
        rel_id_to_label = factory.relation_id_to_label
        
        # convert mapped_triples (IDs) back to string triples
        string_triples = []
        for h, r, t in triples.tolist():
            triple_str = f"{id_to_label[h]}\t{rel_id_to_label[r]}\t{id_to_label[t]}"
            string_triples.append(triple_str)
        
        triple_sets.append(set(string_triples))
        triple_lists.append(string_triples)
    
    # find shared triples
    shared_triples = set.intersection(*triple_sets)
    
    # for each result, get indices of shared triples in original order
    aligned_triples = []
    for string_triples in triple_lists:
        aligned = [triple for triple in string_triples if triple in shared_triples]
        aligned_triples.append(aligned)
    
    return aligned_triples

# this takes in the results from the training step as well as the triples now aligned and grabs the embeddings for them
# I probably could have combined this into the previous function but this was easier to keep track of
def get_embedding_drift_data(aligned_triples, result_array):
    """
    Collect embeddings for aligned triples across datasets.
    
    Returns:
        ```
        {
            "triple_string": {
                "dataset_237": {"head": embedding[], "relation": embedding[], "tail": emebedding[]},
                "dataset_238": {"head": embedding[], "relation": embedding[], "tail": emebedding[]},
                "dataset_239": {"head": emebedding[], "relation": embedding[], "tail": emebedding[]},
            },
            ...
        }
        ```
    """
    drift_data = {}

    for i, aligned in enumerate(aligned_triples):
        dataset_name = f"dataset_23{7+i}"
        result = result_array[i]
        factory = result.training

        # Mappings
        label_to_id_ent = factory.entity_to_id
        label_to_id_rel = factory.relation_to_id

        # Embeddings
        entity_emb = result.model.entity_representations[0]
        relation_emb = result.model.relation_representations[0]

        for triple_str in aligned:
            head_str, rel_str, tail_str = triple_str.split("\t")

            h_id = label_to_id_ent[head_str]
            r_id = label_to_id_rel[rel_str]
            t_id = label_to_id_ent[tail_str]

            device = next(result.model.parameters()).device

            h_emb = entity_emb(torch.tensor(h_id).to(device)).detach().cpu().numpy()
            r_emb = relation_emb(torch.tensor(r_id).to(device)).detach().cpu().numpy()
            t_emb = entity_emb(torch.tensor(t_id).to(device)).detach().cpu().numpy()

            # Store in structured dictionary
            if triple_str not in drift_data:
                drift_data[triple_str] = {}
            
            drift_data[triple_str][dataset_name] = {
                "head": h_emb,
                "relation": r_emb,
                "tail": t_emb,
            }

    return drift_data

# uses the functions to get drift data
def generate_drift_data(num_epochs):
    """
    Runs the full pipeline:
    1. Trains models on datasets fb15k-237, fb15k-238, fb15k-239.
    2. Finds aligned triples shared across datasets.
    3. Collects embeddings for aligned triples.
    4. Returns structured data for further analysis.
    """
    print("Step 1: Training models on datasets...")
    results = train_on_23x(num_epochs)
    
    print("Step 2: Aligning triples across datasets...")
    aligned_triples = sort_results_into_correspondence(results)
    
    print("Step 3: Extracting embeddings for aligned triples...")
    drift_data = get_embedding_drift_data(aligned_triples, results)
    
    print("Embedding drift data collected into memory.")
    
    return drift_data


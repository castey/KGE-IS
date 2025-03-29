import torch
import random
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import os
import numpy as np

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
def train_on_23x():
    
    # going to loop through the datasets and call pipeline() to train on each one
    # then push those results into an array
    # this might make comparing the embeddings simpler too
    result_array = []
    
    for i in range(3):
        x = i + 7
        
        # I'll have to change this eventually to look for the datasets in the proper directory
        # for now this is fine
        DS_PATH = "./ds/fb15k-23" + str(x) 
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

# once we have the triples aligned we can grab the embeddings for those triples
def get_embeddings_for_aligned_triples(aligned_triples, result_array):
    all_embeddings = []

    for i, aligned in enumerate(aligned_triples):
        result = result_array[i]
        factory = result.training

        # Needed mappings
        label_to_id_ent = factory.entity_to_id
        label_to_id_rel = factory.relation_to_id

        # Embedding access (these are torch.nn.Embedding layers)
        entity_emb = result.model.entity_representations[0]
        relation_emb = result.model.relation_representations[0]

        dataset_embeddings = []

        for triple_str in aligned:
            head_str, rel_str, tail_str = triple_str.split("\t")

            h_id = label_to_id_ent[head_str]
            r_id = label_to_id_rel[rel_str]
            t_id = label_to_id_ent[tail_str]

            h_emb = entity_emb(torch.tensor(h_id)).detach().numpy()
            r_emb = relation_emb(torch.tensor(r_id)).detach().numpy()
            t_emb = entity_emb(torch.tensor(t_id)).detach().numpy()

            dataset_embeddings.append((h_emb, r_emb, t_emb))

        all_embeddings.append(dataset_embeddings)

    return all_embeddings  # shape: [dataset][(h_emb, r_emb, t_emb)]

# loop through the embeddings now that they're aligned and represent triples in the intersect of all three datasets
def compare_embeddings(embedding_sets):
    distances = []

    # Number of aligned triples
    num_triples = len(embedding_sets[0])

    for i in range(num_triples):
        triple_distances = {}

        # Get embeddings for this aligned triple across datasets
        emb1 = embedding_sets[0][i]
        emb2 = embedding_sets[1][i]
        emb3 = embedding_sets[2][i]

        # Euclidean distances 
        triple_distances['h_237_238'] = np.linalg.norm(emb1[0] - emb2[0])
        triple_distances['h_237_239'] = np.linalg.norm(emb1[0] - emb3[0])
        triple_distances['h_238_239'] = np.linalg.norm(emb2[0] - emb3[0])

        triple_distances['r_237_238'] = np.linalg.norm(emb1[1] - emb2[1])
        triple_distances['r_237_239'] = np.linalg.norm(emb1[1] - emb3[1])
        triple_distances['r_238_239'] = np.linalg.norm(emb2[1] - emb3[1])

        triple_distances['t_237_238'] = np.linalg.norm(emb1[2] - emb2[2])
        triple_distances['t_237_239'] = np.linalg.norm(emb1[2] - emb3[2])
        triple_distances['t_238_239'] = np.linalg.norm(emb2[2] - emb3[2])

        distances.append(triple_distances)

    return distances

def test_embedding_drift_pipeline():
    # Step 1: Train on each dataset
    print("Training models on datasets...")
    results = train_on_23x()
    
    # Step 2: Get aligned triples across all datasets
    print("Aligning triples across datasets...")
    aligned_triples = sort_results_into_correspondence(results)
    
    # Step 3: Extract embeddings for aligned triples
    print("Extracting embeddings for aligned triples...")
    embedding_sets = get_embeddings_for_aligned_triples(aligned_triples, results)
    
    # Step 4: Compare embeddings (e.g. via Euclidean distance)
    print("Comparing embeddings...")
    distances = compare_embeddings(embedding_sets)
    
    # Step 5: Compute and print averages
    print("Finished comparison. Total aligned triples:", len(distances))
    
    if not distances:
        print("No shared triples to compare.")
        return
    
    # Collect all keys
    keys = distances[0].keys()
    
    # Sum and average
    averages = {key: sum(d[key] for d in distances) / len(distances) for key in keys}
    
    print("Average Euclidean distances across datasets:")
    for k, v in averages.items():
        print(f"{k}: {v:.6f}")

test_embedding_drift_pipeline()
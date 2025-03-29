# Comparing KG Embeddings Independent Study Project

This repo contains three datasets, 237, 238, 239 where 237 ⊂ 238 ⊂ 239

#final.py
final.py uses PyKEEN to train TransE, a KG embedding model on the datasets. It then compares the average euclidean distance between vectors V_n_237 to V_n_238 to V_n_239 to measure drift.

# KGE Drift Code 

Developed using python version 3.11.11

# generate_drift_data.py
generate_drift_data.py contains functions to train on each dataset and returns an object containing each triple which contain embeddings from each dataset `{ triple->dataset->head[], relation[], tail[] }`

The triples present in the final results represent triples that exist in all three datasets.

e.g.
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

# save_drift_data.py
save_drift_data.py calls generate_drift_data.py and saves the results to a json file named drift_data.json

# grab_matrices.py
This script is a demonstration of the deterministic behavior of pykeen when using a fixed seed by calling pipeline() for the 237 dataset with num_epochs=0 then it grabs the matrix and saves to a file. It then does the same thing except with num_epochs set to default.
To set pykeen to deterministic simply set random.seed() and torch.manual_seed() to any number. This seeds all random number generation with a fixed seed ensuring deterministic output. 

# compare_matrices.py
This script compares the outputted files from grab_matrices and shows which are identical and which are different.

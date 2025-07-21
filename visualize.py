# This script uses t-SNE and UMAP to create visualizations of embeddings for:
# FB15k-237, FB15k-238, FB15k-239
# FB15k-238 - FB15k-237, FB15k-239 - FB15k-238, FB15k-239 - FB15k-237
# FB15k-237 using model weights trained on FB15k-238
# FB15k-237 using model weights trained on FB15k-239

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
import torch
import random
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

# Constants
seed = 42
random.seed(seed)
torch.manual_seed(seed)

num_epochs = 100
dataset_paths = {
    "237": "../dataset/fb15k-237",
    "238": "../dataset/fb15k-238",
    "239": "../dataset/fb15k-239"
}

# Create output directory
os.makedirs("img/tsneumap", exist_ok=True)

def get_triples_factories(path):
    return (
        TriplesFactory.from_path(os.path.join(path, "train.txt")),
        TriplesFactory.from_path(os.path.join(path, "valid.txt")),
        TriplesFactory.from_path(os.path.join(path, "test.txt")),
    )

def train_model(train_factory, valid_factory, test_factory, num_epochs=100):
    result = pipeline(
        training=train_factory,
        validation=valid_factory,
        testing=test_factory,
        model="TransE",
        random_seed=seed,
        training_kwargs=dict(num_epochs=num_epochs),
    )
    return result

def get_entity_embeddings(result, entity_list):
    ent2id = result.training.entity_to_id
    emb = result.model.entity_representations[0]()
    indices = [ent2id[e] for e in entity_list]
    return emb[indices].detach().cpu().numpy()

def get_relation_embeddings(result):
    rel_emb = result.model.relation_representations[0]()
    return rel_emb.detach().cpu().numpy()

def transfer_embeddings(source_model, target_model, source_factory, target_factory):
    src_ent2id = source_factory.entity_to_id
    tgt_ent2id = target_factory.entity_to_id
    src_ent_emb = source_model.entity_representations[0]._embeddings.weight.data
    tgt_ent_emb = target_model.entity_representations[0]._embeddings.weight.data

    entity_hits = 0
    for entity, tgt_idx in tgt_ent2id.items():
        if entity in src_ent2id:
            src_idx = src_ent2id[entity]
            tgt_ent_emb[tgt_idx] = src_ent_emb[src_idx]
            entity_hits += 1

    src_rel2id = source_factory.relation_to_id
    tgt_rel2id = target_factory.relation_to_id
    src_rel_emb = source_model.relation_representations[0]._embeddings.weight.data
    tgt_rel_emb = target_model.relation_representations[0]._embeddings.weight.data

    relation_hits = 0
    for relation, tgt_idx in tgt_rel2id.items():
        if relation in src_rel2id:
            src_idx = src_rel2id[relation]
            tgt_rel_emb[tgt_idx] = src_rel_emb[src_idx]
            relation_hits += 1

    print(f"[Info] Transferred {entity_hits}/{len(tgt_ent2id)} entity embeddings, {relation_hits}/{len(tgt_rel2id)} relation embeddings.")

# Visualization functions
def visualize_tsne(entity_embeddings, relation_embeddings, label, num_epochs=100):
    combined = np.vstack([entity_embeddings, relation_embeddings])
    tsne_model = TSNE(n_components=2, random_state=seed, perplexity=30, n_iter=1000)
    combined_2d = tsne_model.fit_transform(combined)

    entities_2d = combined_2d[:len(entity_embeddings)]
    relations_2d = combined_2d[len(entity_embeddings):]

    plt.figure(figsize=(10, 8))
    plt.scatter(entities_2d[:, 0], entities_2d[:, 1],
                c='blue', label='Entities', alpha=0.6, s=10, edgecolor='k')
    if relations_2d.size > 0:
        plt.scatter(relations_2d[:, 0], relations_2d[:, 1],
                    c='red', label='Relations', alpha=0.8, s=30, marker='x')
    plt.title(f't-SNE Visualization: {label}', fontsize=14)
    plt.xlabel('t-SNE Dim 1', fontsize=12)
    plt.ylabel('t-SNE Dim 2', fontsize=12)
    plt.legend(loc='best')
    plt.tight_layout()

    save_path = f"img/tsneumap/{label}_{num_epochs}epochs_tsne.png"
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

def visualize_umap(entity_embeddings, label, num_epochs=100):
    umap_model = umap.UMAP(n_components=2, random_state=seed)
    entity_2d = umap_model.fit_transform(entity_embeddings)

    norm = plt.Normalize(vmin=np.min(entity_2d), vmax=np.max(entity_2d))

    plt.figure(figsize=(10, 8))
    sc = plt.scatter(entity_2d[:, 0], entity_2d[:, 1],
                     c=entity_2d[:, 0], cmap='Spectral',
                     alpha=0.75, s=10, edgecolor='k', norm=norm)
    plt.colorbar(sc, label='Embedding Value')
    plt.title(f'UMAP Visualization: {label}', fontsize=14)
    plt.xlabel('UMAP Dim 1', fontsize=12)
    plt.ylabel('UMAP Dim 2', fontsize=12)
    plt.tight_layout()

    save_path = f"img/tsneumap/{label}_{num_epochs}epochs_umap.png"
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()
    print(f"[Saved] {save_path}")

# === Train original datasets ===
results = []
for key in ["237", "238", "239"]:
    train, valid, test = get_triples_factories(dataset_paths[key])
    results.append(train_model(train, valid, test, num_epochs=num_epochs))

# === Train fresh models for 237 (to transfer weights into) ===
train_237, valid_237, test_237 = get_triples_factories(dataset_paths["237"])
results.append(train_model(train_237, valid_237, test_237, num_epochs=0))  # index 3
results.append(train_model(train_237, valid_237, test_237, num_epochs=0))  # index 4

# === Transfer model weights carefully ===
transfer_embeddings(results[1].model, results[3].model, results[1].training, results[3].training)  # 238 --> fresh 237
transfer_embeddings(results[2].model, results[4].model, results[2].training, results[4].training)  # 239 --> fresh 237

# === Prepare Sets ===
def get_common_entities(results):
    all_sets = [set(r.training.entity_to_id.keys()) for r in results]
    return sorted(set.intersection(*all_sets))

common_entities = get_common_entities(results)

# Common sets (aligned entities)
entity_embeddings_237 = get_entity_embeddings(results[0], common_entities)
entity_embeddings_238 = get_entity_embeddings(results[1], common_entities)
entity_embeddings_239 = get_entity_embeddings(results[2], common_entities)
entity_embeddings_T237_M238 = get_entity_embeddings(results[3], common_entities)
entity_embeddings_T237_M239 = get_entity_embeddings(results[4], common_entities)

relation_embeddings_237 = get_relation_embeddings(results[0])
relation_embeddings_238 = get_relation_embeddings(results[1])
relation_embeddings_239 = get_relation_embeddings(results[2])
relation_embeddings_T237_M238 = get_relation_embeddings(results[3])
relation_embeddings_T237_M239 = get_relation_embeddings(results[4])

# Difference sets (new entities only)
def get_entity_set(result):
    return set(result.training.entity_to_id.keys())

entities_237 = get_entity_set(results[0])
entities_238 = get_entity_set(results[1])
entities_239 = get_entity_set(results[2])

entities_238_minus_237 = sorted(entities_238 - entities_237)
entities_239_minus_238 = sorted(entities_239 - entities_238)
entities_239_minus_237 = sorted(entities_239 - entities_237)

entity_embeddings_238_minus_237 = get_entity_embeddings(results[1], entities_238_minus_237)
entity_embeddings_239_minus_238 = get_entity_embeddings(results[2], entities_239_minus_238)
entity_embeddings_239_minus_237 = get_entity_embeddings(results[2], entities_239_minus_237)

# Final list
sets = [
    (entity_embeddings_237, relation_embeddings_237, "fb15k237"),
    (entity_embeddings_238, relation_embeddings_238, "fb15k238"),
    (entity_embeddings_239, relation_embeddings_239, "fb15k239"),
    (entity_embeddings_238_minus_237, np.zeros((0, entity_embeddings_238_minus_237.shape[1])), "fb15k238_minus_237"),
    (entity_embeddings_239_minus_238, np.zeros((0, entity_embeddings_239_minus_238.shape[1])), "fb15k239_minus_238"),
    (entity_embeddings_239_minus_237, np.zeros((0, entity_embeddings_239_minus_237.shape[1])), "fb15k239_minus_237"),
    (entity_embeddings_T237_M238, relation_embeddings_T237_M238, "t237_m238"),
    (entity_embeddings_T237_M239, relation_embeddings_T237_M239, "t237_m239"),
]

# === Visualize all sets
for entity_embs, relation_embs, label in sets:
    visualize_tsne(entity_embs, relation_embs, label, num_epochs)
    visualize_umap(entity_embs, label, num_epochs)

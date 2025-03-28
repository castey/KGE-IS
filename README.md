# Comparing KG Embeddings Independent Study Project

This repo contains three datasets, 237, 238, 239 where 237 ⊂ 238 ⊂ 239

final.py uses PyKEEN to train a TransE embedding model on the datasets. It then compares the average euclidean distance between vectors V_n_237 to V_n_238 to V_n_239 to measure drift.


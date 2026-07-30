import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_dataset

EMBEDDING_DIR = "embeddings"

os.makedirs(EMBEDDING_DIR, exist_ok=True)

print("Loading dataset...")
documents = load_dataset()

texts = [doc["text"] for doc in documents]

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Generating embeddings...")
vectors = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
)

print("Saving vectors...")
np.save(f"{EMBEDDING_DIR}/vectors.npy", vectors)

print("Saving metadata...")
with open(f"{EMBEDDING_DIR}/metadata.json", "w") as f:
    json.dump(documents, f, indent=2)

print("Done!")
print(f"Documents: {len(documents)}")
print(f"Embedding Dimension: {vectors.shape[1]}")
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    SearchParams,
    VectorParams,
)

client = QdrantClient("http://localhost:6333")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

vectors = np.load("embeddings/vectors.npy")

with open("embeddings/metadata.json") as f:
    metadata = json.load(f)

# ------------------------------------------------
# Create deliberately poor HNSW collection
# ------------------------------------------------

collection = "documents_hnsw_low"

if client.collection_exists(collection):
    client.delete_collection(collection)

client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
    hnsw_config=HnswConfigDiff(
        m=8,
        ef_construct=32,
    ),
)

points = []

for idx, (vector, doc) in enumerate(zip(vectors, metadata)):
    points.append(
        {
            "id": idx,
            "vector": vector.tolist(),
            "payload": doc,
        }
    )

BATCH_SIZE = 200

for i in range(0, len(points), BATCH_SIZE):
    client.upsert(
        collection_name=collection,
        points=points[i:i + BATCH_SIZE],
    )

queries = [
    "How do I fix a graphics card driver issue?",
    "Why won't my computer boot after installing new hardware?",
    "Which baseball team had the best pitching season?",
    "What causes a space shuttle launch delay?",
    "How can I improve motorcycle engine performance?",
]

results = {}

for query in queries:

    query_vector = model.encode(query).tolist()

    results[query] = {}

    # ----------------------------
    # Exact Search
    # ----------------------------

    start = time.perf_counter()

    exact = client.query_points(
        collection_name="documents_cosine",
        query=query_vector,
        search_params=SearchParams(
            exact=True
        ),
        limit=5,
    )

    exact_latency = (
        time.perf_counter() - start
    ) * 1000

    exact_ids = [
        point.id
        for point in exact.points
    ]

    results[query]["exact"] = {
        "latency_ms": exact_latency,
        "ids": exact_ids,
    }

    # ----------------------------
    # Default HNSW
    # ----------------------------

    start = time.perf_counter()

    default = client.query_points(
        collection_name="documents_cosine",
        query=query_vector,
        limit=5,
    )

    latency = (
        time.perf_counter() - start
    ) * 1000

    default_ids = [
        point.id
        for point in default.points
    ]

    overlap = len(
        set(default_ids) & set(exact_ids)
    )

    results[query]["default_hnsw"] = {
        "latency_ms": latency,
        "overlap": overlap,
    }

    # ----------------------------
    # Poor HNSW
    # ----------------------------

    for ef in [16, 64, 128]:

        start = time.perf_counter()

        search = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=5,
            search_params=SearchParams(
                hnsw_ef=ef
            ),
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        ids = [
            point.id
            for point in search.points
        ]

        overlap = len(
            set(ids) & set(exact_ids)
        )

        results[query][f"poor_hnsw_ef_{ef}"] = {
            "latency_ms": latency,
            "overlap": overlap,
        }

with open(
    "results/hnsw.json",
    "w",
) as f:
    json.dump(results, f, indent=2)

print("Finished.")
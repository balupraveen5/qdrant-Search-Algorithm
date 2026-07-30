import json
import uuid

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

# ----------------------------
# Connect to Qdrant
# ----------------------------

client = QdrantClient(url="http://localhost:6333")

# ----------------------------
# Load saved embeddings
# ----------------------------

vectors = np.load("embeddings/vectors.npy")

with open("embeddings/metadata.json") as f:
    metadata = json.load(f)

# ----------------------------
# Embedding model
# ----------------------------

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

collections = {
    "documents_cosine": Distance.COSINE,
    "documents_euclid": Distance.EUCLID,
    "documents_dot": Distance.DOT,
}

# ----------------------------
# Create collections
# ----------------------------

for name, distance in collections.items():

    if client.collection_exists(name):
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=384,
            distance=distance,
        ),
    )

    points = []

    for vector, doc in zip(vectors, metadata):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={
                    "text": doc["text"],
                    "category": doc["category"],
                },
            )
        )

    BATCH_SIZE = 200
    for i in range(0, len(points), BATCH_SIZE):
            client.upsert(
            collection_name=name,
            points=points[i:i + BATCH_SIZE],
              )

    print(f"Uploaded {len(points)} points to {name}")

# ----------------------------
# Test Queries
# ----------------------------

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

    for collection in collections:

        search = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=5,
            with_payload=True,
        )

        top5 = []

        for point in search.points:

            top5.append(
                {
                    "score": point.score,
                    "category": point.payload["category"],
                    "text": point.payload["text"][:120],
                }
            )

        results[query][collection] = top5

with open(
    "results/distance_metrics.json",
    "w",
) as f:
    json.dump(results, f, indent=2)

print("Results saved.")
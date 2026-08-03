import json
import time

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load vectors
vectors = np.load("embeddings/vectors.npy")

with open("embeddings/metadata.json") as f:
    metadata = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Build IVF Index
# -----------------------------
k_clusters = 32

kmeans = KMeans(n_clusters=k_clusters)
labels = kmeans.fit_predict(vectors)

centroids = kmeans.cluster_centers_

# Store vector ids for each cluster
clusters = {}

for i, label in enumerate(labels):
    clusters.setdefault(label, []).append(i)


# -----------------------------
# IVF Search
# -----------------------------
def search_ivf(query_vector, nprobe=1, top_k=5):

    # Find nearest centroids
    centroid_scores = cosine_similarity(
        [query_vector],
        centroids
    )[0]

    nearest_clusters = np.argsort(centroid_scores)[::-1][:nprobe]

    candidate_ids = []

    for cluster in nearest_clusters:
        candidate_ids.extend(clusters[cluster])

    results = []

    for idx in candidate_ids:

        score = cosine_similarity(
            [query_vector],
            [vectors[idx]]
        )[0][0]

        results.append(
            {
                "id": idx,
                "score": float(score),
                "category": metadata[idx]["category"],
                "text": metadata[idx]["text"][:100],
            }
        )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return results[:top_k]


# -----------------------------
# Test Queries
# -----------------------------

queries = [
    "How do I fix a graphics card driver issue?",
    "Why won't my computer boot after installing new hardware?",
    "Which baseball team had the best pitching season?",
    "What causes a space shuttle launch delay?",
    "How can I improve motorcycle engine performance?",
]

output = {}

for query in queries:

    query_vector = model.encode(query)

    output[query] = {}

    for nprobe in [1, 8]:

        start = time.perf_counter()

        results = search_ivf(
            query_vector,
            nprobe=nprobe,
            top_k=5,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        output[query][f"nprobe_{nprobe}"] = {
            "latency_ms": round(latency, 2),
            "results": results,
        }

        print(
            f"{query} | nprobe={nprobe} | {latency:.2f} ms"
        )

with open(
    "results/ivf.json",
    "w",
) as f:
    json.dump(output, f, indent=2)

print("\nIVF results saved.")
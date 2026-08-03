## Observations

### Cosine Similarity

Cosine similarity consistently returned documents that were semantically relevant to each query. It correctly matched topics such as graphics drivers, computer hardware, baseball, space exploration, and motorcycle performance. Higher cosine scores indicated stronger semantic similarity.

### Euclidean Distance

Euclidean distance retrieved the same documents as cosine similarity for all tested queries. The difference was only in score interpretation: smaller distance values indicate closer vectors. Since the document rankings remained unchanged, Euclidean proved equally effective for this normalized embedding dataset.

### Dot Product

Dot product produced identical rankings and nearly identical scores to cosine similarity. This is expected because the SentenceTransformer embeddings are normalized (or effectively normalized), making cosine similarity and dot product mathematically equivalent for retrieval.


# HNSW vs Exact Search Comparison

| Method | Search Parameter | Avg. Top-5 Overlap (out of 5) | Avg. Latency (ms) |
|--------|------------------|------------------------------:|------------------:|
| Exact (Brute Force) | — | 5.0 (Ground Truth) | 2.857 |
| Default HNSW | Default | 5.0 | 2.162 |
| Poor HNSW | ef = 16 | 5.0 | 2.073 |
| Poor HNSW | ef = 64 | 5.0 | 1.937 |
| Poor HNSW | ef = 128 | 5.0 | 1.946 |
| IVF | nprobe = 1 | 4.0 | 16.456 |
| IVF | nprobe = 8 | 5.0 | 133.614 |
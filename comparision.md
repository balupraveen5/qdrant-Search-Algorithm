## Observations

### Cosine Similarity

Cosine similarity consistently returned documents that were semantically relevant to each query. It correctly matched topics such as graphics drivers, computer hardware, baseball, space exploration, and motorcycle performance. Higher cosine scores indicated stronger semantic similarity.

### Euclidean Distance

Euclidean distance retrieved the same documents as cosine similarity for all tested queries. The difference was only in score interpretation: smaller distance values indicate closer vectors. Since the document rankings remained unchanged, Euclidean proved equally effective for this normalized embedding dataset.

### Dot Product

Dot product produced identical rankings and nearly identical scores to cosine similarity. This is expected because the SentenceTransformer embeddings are normalized (or effectively normalized), making cosine similarity and dot product mathematically equivalent for retrieval.


# HNSW vs Exact Search Comparison

| Query | Exact Latency (ms) | Default HNSW Latency (ms) | Top-5 Overlap | Poor HNSW (ef=16) Overlap | Poor HNSW (ef=64) Overlap | Poor HNSW (ef=128) Overlap |
|-------|-------------------:|--------------------------:|--------------:|--------------------------:|--------------------------:|---------------------------:|
| How do I fix a graphics card driver issue? | 14.60 | 3.48 | 5/5 | 0/5 | 0/5 | 0/5 |
| Why won't my computer boot after installing new hardware? | 3.28 | 2.72 | 5/5 | 0/5 | 0/5 | 0/5 |
| Which baseball team had the best pitching season? | 4.25 | 3.04 | 5/5 | 0/5 | 0/5 | 0/5 |
| What causes a space shuttle launch delay? | 5.34 | 3.26 | 5/5 | 0/5 | 0/5 | 0/5 |
| How can I improve motorcycle engine performance? | 4.58 | 2.39 | 5/5 | 0/5 | 0/5 | 0/5 |

## Summary

- **Exact search** always returns the ground-truth nearest neighbors but has higher latency (3.28–14.60 ms).
- **Default HNSW** is significantly faster (2.39–3.48 ms) while achieving **100% Top-5 overlap (5/5)** with exact search for all test queries.
- **Poorly configured HNSW** (low-quality index with varying `ef`) is slightly faster but fails to retrieve any of the exact Top-5 neighbors (**0/5 overlap**), demonstrating the trade-off between search quality and index configuration.
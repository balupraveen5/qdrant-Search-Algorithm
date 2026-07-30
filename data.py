from sklearn.datasets import fetch_20newsgroups


def load_dataset(limit=6000):
    """
    Load a subset of the 20 Newsgroups dataset.
    Returns a list of dictionaries with text and label.
    """
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )

    documents = []

    for text, target in zip(dataset.data[:limit], dataset.target[:limit]):
        cleaned = text.strip()

        if cleaned:
            documents.append(
                {
                    "text": cleaned,
                    "category": dataset.target_names[target],
                }
            )

    return documents


if __name__ == "__main__":
    print("Started")
    docs = load_dataset()

    print(f"Loaded {len(docs)} documents")
    print(docs[0])
    print("end")
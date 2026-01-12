import pickle
import gc
import torch
from tqdm import tqdm
from langchain_community.vectorstores import FAISS

from src.config import SPLITS_PATH, INDEX_PATH
from src.models import get_embeddings
from src.utils import cleanup_memory


def build_vector_index(batch_size: int = 20) -> None:
    """
    Loads processed splits and builds a FAISS index in batches to avoid OOM.

    Args:
        batch_size (int): Number of documents to process before clearing CUDA cache.
    """
    print(f"Loading splits from {SPLITS_PATH}...")
    try:
        with open(SPLITS_PATH, "rb") as f:
            new_splits = pickle.load(f)
    except FileNotFoundError:
        print("Error: Splits file not found. Run ingestion first.")
        return

    print(f"Total documents to index: {len(new_splits)}")

    embeddings = get_embeddings()

    # Create empty index first or process first batch
    print("Initializing FAISS index with first batch...")
    first_batch = new_splits[:batch_size]
    vector_db = FAISS.from_documents(first_batch, embeddings)

    # Process remaining batches
    for i in tqdm(range(batch_size, len(new_splits), batch_size), desc="Indexing Batches"):
        try:
            batch = new_splits[i : i + batch_size]
            vector_db.add_documents(batch)

            # Memory Management from original script
            cleanup_memory()

            # Periodic Save (optional safety)
            if i % 500 == 0:
                vector_db.save_local(str(INDEX_PATH))

        except Exception as e:
            print(f"Error indexing batch {i}: {e}")
            # Recovery logic from original script
            vector_db.save_local(str(INDEX_PATH))
            del vector_db
            cleanup_memory()

            # Reload
            vector_db = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
            # Retry batch item by item
            for item in batch:
                vector_db.add_documents([item])

    print(f"Saving final index to {INDEX_PATH}...")
    vector_db.save_local(str(INDEX_PATH))
    print("Indexing complete.")

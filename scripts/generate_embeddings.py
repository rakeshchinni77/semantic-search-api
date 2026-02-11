import os
import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# Configuration

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "models/sentence_transformer")
DATA_DIR = os.getenv("DATA_DIR", "data")

DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")

NUM_DOCUMENTS = int(os.getenv("NUM_DOCUMENTS", "1000"))


# Logging Setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Utility Functions

def ensure_directories() -> None:
    Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def generate_synthetic_documents(num_docs: int) -> List[Dict[str, str]]:
    topics = [
        "machine learning",
        "artificial intelligence",
        "natural language processing",
        "deep learning",
        "data engineering",
        "cloud computing",
        "finance analytics",
        "healthcare systems",
        "e-commerce platforms",
        "cybersecurity"
    ]

    documents = []
    for i in range(num_docs):
        topic = topics[i % len(topics)]
        documents.append({
            "id": f"doc_{i+1}",
            "text": (
                f"This document discusses concepts related to {topic}. "
                f"It provides insights and practical examples in {topic} applications."
            )
        })

    return documents


def save_documents(documents: List[Dict[str, str]]) -> None:
    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=4)
    logger.info(f"Saved {len(documents)} documents to {DOCUMENTS_FILE}")


def generate_embeddings(model: SentenceTransformer, documents: List[Dict[str, str]]) -> np.ndarray:
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings.astype("float32")  # Required for FAISS


def save_embeddings(embeddings: np.ndarray) -> None:
    np.save(EMBEDDINGS_FILE, embeddings)
    logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")


def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Create FAISS IndexFlatL2 and add embeddings.
    """
    dimension = embeddings.shape[1]
    logger.info(f"Initializing FAISS IndexFlatL2 with dimension {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Added {index.ntotal} vectors to FAISS index")
    return index


def save_faiss_index(index: faiss.Index) -> None:
    faiss.write_index(index, FAISS_INDEX_FILE)
    logger.info(f"Saved FAISS index to {FAISS_INDEX_FILE}")


# Main Execution

def main() -> None:
    try:
        logger.info("Starting embedding and FAISS index generation process...")

        ensure_directories()

        logger.info(f"Loading model: {MODEL_NAME}")
        model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=MODEL_CACHE_DIR
        )

        logger.info(f"Generating {NUM_DOCUMENTS} synthetic documents...")
        documents = generate_synthetic_documents(NUM_DOCUMENTS)
        save_documents(documents)

        logger.info("Generating embeddings...")
        embeddings = generate_embeddings(model, documents)
        save_embeddings(embeddings)

        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        save_faiss_index(index)

        logger.info("Embedding and FAISS indexing completed successfully.")

    except Exception as e:
        logger.error(f"Error during embedding/index generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()

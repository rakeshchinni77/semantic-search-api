import os
import json
import logging
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load Environment Variables

load_dotenv()


# Logging Configuration

logger = logging.getLogger(__name__)


# Custom Exceptions

class SearchServiceInitializationError(Exception):
    """Raised when the service fails during initialization."""


class SearchServiceQueryError(Exception):
    """Raised when query processing fails."""


# Search Service

class SearchService:
    """
    Business logic layer for semantic search.
    Responsible for:
    - Loading embedding model
    - Loading FAISS index
    - Loading document store
    - Performing similarity search
    """

    def __init__(self):
        try:
            logger.info("Initializing SearchService...")

           # Environment Configuration

            self.model_name = os.getenv("MODEL_NAME")
            self.model_cache_dir = os.getenv("MODEL_CACHE_DIR")
            self.data_dir = os.getenv("DATA_DIR")
            self.documents_file = os.getenv("DOCUMENTS_FILE")
            self.faiss_index_file = os.getenv("FAISS_INDEX_FILE")

            if not all([
                self.model_name,
                self.model_cache_dir,
                self.data_dir,
                self.documents_file,
                self.faiss_index_file
            ]):
                raise ValueError("Missing required environment variables.")

            self.documents_path = os.path.join(self.data_dir, self.documents_file)
            self.faiss_index_path = os.path.join(self.data_dir, self.faiss_index_file)

            
            # Load Components
            
            self.model = self._load_model()
            self.documents = self._load_documents()
            self.index = self._load_faiss_index()

            logger.info("SearchService initialized successfully.")

        except Exception as e:
            logger.exception("Failed to initialize SearchService.")
            raise SearchServiceInitializationError(str(e))

    # Private Loaders

    def _load_model(self) -> SentenceTransformer:
        logger.info(f"Loading model: {self.model_name}")
        return SentenceTransformer(
            self.model_name,
            cache_folder=self.model_cache_dir
        )

    def _load_documents(self) -> List[Dict]:
        if not os.path.exists(self.documents_path):
            raise FileNotFoundError(
                f"Documents file not found at {self.documents_path}"
            )

        with open(self.documents_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        logger.info(f"Loaded {len(documents)} documents.")
        return documents

    def _load_faiss_index(self) -> faiss.Index:
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(
                f"FAISS index file not found at {self.faiss_index_path}"
            )

        index = faiss.read_index(self.faiss_index_path)

        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")

        logger.info(f"Loaded FAISS index with {index.ntotal} vectors.")
        return index

    # Public Search Method

    def search_documents(self, query: str, top_k: int) -> List[Dict]:
        """
        Perform semantic search against FAISS index.
        """

        if not query or not query.strip():
            raise SearchServiceQueryError("Query must not be empty.")

        if top_k <= 0:
            raise SearchServiceQueryError("top_k must be greater than zero.")

        try:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")

            distances, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(distances[0], indices[0]):

                if idx < 0 or idx >= len(self.documents):
                    continue

                document = self.documents[idx]

                results.append({
                    "id": document["id"],
                    "text_snippet": document["text"][:200],
                    "score": float(score)
                })

            return results

        except Exception as e:
            logger.exception("Error during search operation.")
            raise SearchServiceQueryError(str(e))

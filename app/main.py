import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.search_service import (
    SearchService,
    SearchServiceInitializationError,
    SearchServiceQueryError,
)


# Environment Setup

load_dotenv()

API_TITLE = os.getenv("API_TITLE")
API_VERSION = os.getenv("API_VERSION")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

if not API_TITLE or not API_VERSION:
    raise ValueError("Missing required API environment variables.")


# Logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI App

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION
)



# Singleton Service Instance

_search_service_instance: SearchService | None = None


def get_search_service() -> SearchService:
    """
    Dependency injection for SearchService.
    Ensures singleton pattern (model/index loaded only once).
    """
    global _search_service_instance

    if _search_service_instance is None:
        try:
            _search_service_instance = SearchService()
        except SearchServiceInitializationError as e:
            logger.exception("SearchService failed to initialize.")
            raise HTTPException(
                status_code=500,
                detail="Search service initialization failed."
            )

    return _search_service_instance


# Pydantic Models

class SearchRequest(BaseModel):
    query: str


class SearchResultItem(BaseModel):
    id: str
    text_snippet: str
    score: float


# Health Endpoint

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}


# Search Endpoint

@app.post(
    "/search",
    response_model=List[SearchResultItem],
    status_code=200
)
async def semantic_search_endpoint(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service)
):
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty and at least 3 characters long."
        )

    try:
        results = service.search_documents(
            query=request.query,
            top_k=TOP_K_RESULTS
        )
        return results

    except SearchServiceQueryError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception:
        logger.exception("Unexpected error during search.")
        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )

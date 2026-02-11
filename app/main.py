import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.search_service import (
    SearchService,
    SearchServiceInitializationError,
    SearchServiceQueryError,
    SearchServiceResourceNotFoundError,
)


# Load Environment Variables

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


# Singleton Instance

_search_service_instance: SearchService | None = None


def get_search_service() -> SearchService:
    global _search_service_instance

    if _search_service_instance is None:
        _search_service_instance = SearchService()

    return _search_service_instance


# Exception Handlers

@app.exception_handler(SearchServiceInitializationError)
async def initialization_exception_handler(request: Request, exc: SearchServiceInitializationError):
    return JSONResponse(
        status_code=500,
        content={"detail": "Search service initialization failed."}
    )


@app.exception_handler(SearchServiceResourceNotFoundError)
async def resource_not_found_handler(request: Request, exc: SearchServiceResourceNotFoundError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )


@app.exception_handler(SearchServiceQueryError)
async def query_exception_handler(request: Request, exc: SearchServiceQueryError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred.")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."}
    )


# Pydantic Models

class SearchRequest(BaseModel):
    query: str


class SearchResultItem(BaseModel):
    id: str
    text_snippet: str
    score: float


# Endpoints

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}


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

    return service.search_documents(
        query=request.query,
        top_k=TOP_K_RESULTS
    )

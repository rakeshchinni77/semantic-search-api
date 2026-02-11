import os
import pytest

from app.services.search_service import (
    SearchService,
    SearchServiceQueryError,
)


@pytest.fixture(scope="module")
def search_service():
    """
    Create a SearchService instance once for unit tests.
    """
    return SearchService()


def test_service_initialization(search_service):
    """
    Ensure model, documents, and index are loaded.
    """
    assert search_service.model is not None
    assert search_service.documents is not None
    assert len(search_service.documents) > 0
    assert search_service.index.ntotal > 0


def test_search_returns_list(search_service):
    """
    Ensure search_documents returns a list.
    """
    results = search_service.search_documents("machine learning", top_k=5)
    assert isinstance(results, list)
    assert len(results) == 5


def test_search_result_format(search_service):
    """
    Ensure result items contain required keys.
    """
    results = search_service.search_documents("machine learning", top_k=3)

    for item in results:
        assert "id" in item
        assert "text_snippet" in item
        assert "score" in item
        assert isinstance(item["id"], str)
        assert isinstance(item["text_snippet"], str)
        assert isinstance(item["score"], float)


def test_empty_query_raises_error(search_service):
    """
    Ensure empty query raises SearchServiceQueryError.
    """
    with pytest.raises(SearchServiceQueryError):
        search_service.search_documents("", top_k=5)


def test_invalid_top_k_raises_error(search_service):
    """
    Ensure invalid top_k raises SearchServiceQueryError.
    """
    with pytest.raises(SearchServiceQueryError):
        search_service.search_documents("machine learning", top_k=0)

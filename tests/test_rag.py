"""Unit and integration test suite for RouteOps RAG Store.

Tests document indexing, NumPy cosine similarity retrieval, category filtering,
top-k ranking, and snippet generation across eSewa, Khalti, NTC, Ncell, local banking,
and API rate limit documentation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rag_store import CURATED_DOMAIN_DOCS, RAGDocument, RAGStore


@pytest.fixture
def rag_store() -> RAGStore:
    """Fixture providing an initialized RAGStore instance."""
    return RAGStore()


class TestRAGStoreInitialization:
    """Tests for RAGStore setup and document index construction."""

    def test_document_count(self, rag_store: RAGStore) -> None:
        """Verify RAGStore loads 25-30 curated documents."""
        assert len(rag_store.documents) >= 25
        assert len(rag_store.documents) == len(CURATED_DOMAIN_DOCS)
        assert rag_store.embeddings.shape[0] == len(rag_store.documents)
        assert rag_store.embeddings.shape[1] == 128

    def test_embedding_normalization(self, rag_store: RAGStore) -> None:
        """Verify all document embeddings are L2 normalized unit vectors."""
        for i in range(rag_store.embeddings.shape[0]):
            norm = float(np.linalg.norm(rag_store.embeddings[i]))
            assert np.isclose(norm, 1.0, atol=1e-4)


class TestRAGStoreRetrieval:
    """Tests for semantic search and retrieval accuracy across knowledge domains."""

    def test_search_esewa_transaction_query(self, rag_store: RAGStore) -> None:
        """Verify query about eSewa pending transactions retrieves esewa_001 top."""
        query = "Mero eSewa transaction pending bhayo, paisa katyo, k garne?"
        results = rag_store.search(query, top_k=3)

        assert len(results) > 0
        top_doc, top_score = results[0]
        assert top_doc.category == "esewa_khalti"
        assert "esewa_001" in top_doc.doc_id
        assert top_score > 0.30

    def test_search_khalti_mpin_query(self, rag_store: RAGStore) -> None:
        """Verify query about Khalti MPIN lock retrieves khalt_002 or relevant doc."""
        query = "Khalti MPIN wrong halda wallet lock bhayo, kasari unlock garne?"
        results = rag_store.search(query, top_k=3)

        assert len(results) > 0
        doc_ids = [d.doc_id for d, _ in results]
        assert "esewa_002" in doc_ids

    def test_search_ntc_data_pack(self, rag_store: RAGStore) -> None:
        """Verify query about NTC data pack activation retrieves NTC documents."""
        query = "Nepal Telecom NTC ko 1GB 24 hour data pack kasari kinne code k ho?"
        results = rag_store.search(query, top_k=3, category="ntc_ncell")

        assert len(results) > 0
        top_doc, top_score = results[0]
        assert top_doc.category == "ntc_ncell"
        assert "ntc_001" in top_doc.doc_id

    def test_search_bank_fixed_deposit(self, rag_store: RAGStore) -> None:
        """Verify query about fixed deposit interest rates retrieves banking docs."""
        query = "Commercial bank haru ma fixed deposit FD ko interest rate kati chha?"
        results = rag_store.search(query, top_k=3, category="banking_kyc")

        assert len(results) > 0
        top_doc, top_score = results[0]
        assert top_doc.category == "banking_kyc"
        assert "bank_001" in top_doc.doc_id

    def test_search_api_rate_limits(self, rag_store: RAGStore) -> None:
        """Verify query about API rate limits retrieves RouteOps API docs."""
        query = "What is the REST API rate limit and 429 too many requests policy?"
        results = rag_store.search(query, top_k=3, category="api_limits")

        assert len(results) > 0
        top_doc, _ = results[0]
        assert top_doc.category == "api_limits"
        assert "api_001" in top_doc.doc_id


class TestRAGStoreFilteringAndDynamicAddition:
    """Tests for category filtering and adding custom dynamic documents."""

    def test_category_filtering(self, rag_store: RAGStore) -> None:
        """Verify search with category filter only returns docs from that category."""
        results = rag_store.search("balance check", top_k=5, category="ntc_ncell")
        for doc, _ in results:
            assert doc.category == "ntc_ncell"

    def test_add_dynamic_document(self, rag_store: RAGStore) -> None:
        """Verify adding a new document updates index and makes it immediately searchable."""
        initial_count = len(rag_store.documents)
        rag_store.add_document(
            doc_id="custom_099",
            title="Custom RouteOps Multilingual Routing Protocol",
            category="api_limits",
            content="RouteOps classifies queries into 4 semantic classes with sub-millisecond n-gram projection.",
            keywords=["routeops", "multilingual", "routing"],
        )

        assert len(rag_store.documents) == initial_count + 1
        assert rag_store.embeddings.shape[0] == initial_count + 1

        results = rag_store.search("RouteOps multilingual routing protocol", top_k=1)
        assert len(results) == 1
        doc, _ = results[0]
        assert doc.doc_id == "custom_099"

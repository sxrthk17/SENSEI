
import pytest
from langchain_core.documents import Document

from rag_pipeline import (
    _build_filter,
    _matches_scope,
    hybrid_search,
    keyword_score,
)


class FakeCollection:
    def __init__(self, documents):
        self.documents = documents

    def get(self, where=None, include=None):
        matched = []

        for doc in self.documents:
            metadata = doc.metadata or {}

            if where and not self._matches_filter(metadata, where):
                continue

            matched.append(doc)

        return {
            "documents": [doc.page_content for doc in matched],
            "metadatas": [doc.metadata for doc in matched],
        }

    @staticmethod
    def _matches_filter(metadata, where):
        for key, expected in where.items():
            actual = metadata.get(key)

            if isinstance(expected, dict):
                if "$in" in expected:
                    if actual not in expected["$in"]:
                        return False
                elif "$eq" in expected:
                    if actual != expected["$eq"]:
                        return False
                else:
                    return False
            elif actual != expected:
                return False

        return True


class FakeVectorStore:
    def __init__(self, documents):
        self._collection = FakeCollection(documents)

    def similarity_search(self, query, k=5, filter=None):
        result = self._collection.get(where=filter)

        docs = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                result["documents"],
                result["metadatas"],
            )
        ]

        docs.sort(
            key=lambda doc: keyword_score(query, doc.page_content),
            reverse=True,
        )
        return docs[:k]


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content=(
                "Gauge factor is the ratio of fractional "
                "resistance change to strain."
            ),
            metadata={
                "document_id": "mdm-1",
                "chunk_index": 0,
                "source_pdf": "Numerical MDM.pdf",
                "subject": "MDM",
            },
        ),
        Document(
            page_content=(
                "An RTD measures temperature through "
                "resistance variation."
            ),
            metadata={
                "document_id": "mdm-1",
                "chunk_index": 1,
                "source_pdf": "Numerical MDM.pdf",
                "subject": "MDM",
            },
        ),
        Document(
            page_content=(
                "Backpropagation computes gradients "
                "in neural networks."
            ),
            metadata={
                "document_id": "ai-1",
                "chunk_index": 0,
                "source_pdf": "AI Notes.pdf",
                "subject": "AI",
            },
        ),
    ]


def test_keyword_score_rewards_matching_terms():
    relevant = keyword_score(
        "gauge factor strain",
        "Gauge factor depends on strain.",
    )
    irrelevant = keyword_score(
        "gauge factor strain",
        "Database normalization.",
    )

    assert relevant > irrelevant


def test_build_filter_for_document_id():
    result = _build_filter(document_id="mdm-1")

    # Chroma filter format used by the actual implementation.
    assert result == {
        "document_id": {"$in": ["mdm-1"]}
    }


def test_matches_scope_accepts_matching_document(sample_docs):
    metadata = sample_docs[0].metadata

    assert _matches_scope(
        metadata,
        document_ids=["mdm-1"],
    )


def test_matches_scope_rejects_other_document(sample_docs):
    metadata = sample_docs[2].metadata

    assert not _matches_scope(
        metadata,
        document_ids=["mdm-1"],
    )


def test_hybrid_search_returns_results_within_scope(sample_docs):
    store = FakeVectorStore(sample_docs)

    results = hybrid_search(
        store,
        "gauge factor strain",
        k=5,
        where={"document_id": {"$in": ["mdm-1"]}},
    )

    assert results
    assert all(
        doc.metadata["document_id"] == "mdm-1"
        for doc in results
    )


def test_hybrid_search_fails_closed_without_scope(sample_docs):
    store = FakeVectorStore(sample_docs)

    assert hybrid_search(
        store,
        "gauge factor",
        where=None,
    ) == []


def test_hybrid_search_respects_requested_limit(sample_docs):
    store = FakeVectorStore(sample_docs)

    results = hybrid_search(
        store,
        "temperature resistance",
        k=1,
        where={"document_id": {"$in": ["mdm-1"]}},
    )

    assert len(results) <= 1

from rag_pipeline import generate_document_id, document_already_ingested


def test_document_id_is_stable_for_same_pdf(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"test PDF content")

    first_id = generate_document_id(str(pdf))
    second_id = generate_document_id(str(pdf))

    assert first_id == second_id


def test_document_id_changes_when_pdf_content_changes(tmp_path):
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"first PDF content")
    first_id = generate_document_id(str(pdf))

    pdf.write_bytes(b"changed PDF content")
    second_id = generate_document_id(str(pdf))

    assert first_id != second_id

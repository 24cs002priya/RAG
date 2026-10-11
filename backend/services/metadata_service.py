"""Metadata normalization and MongoDB filter helpers."""

'''may adapt from:ingestion.py, schemas.py'''
'''for example {
    "document_id": "...",
    "source": "market_report.pdf",
    "page": 15,
    "chunk_id": "15_03",
    "chunk_type": "text",
    "document_type": "research_report",
    "industry": "automotive",
    "client": "ABC",
    "project": "EV Market Study",
    "year": 2025
}'''
'''Not every field has to be available for every document.

Concept applied

Metadata Architecture

Result

Every chunk can be identified and filtered.

For example:

document_type = research_report
industry = automotive
year = 2025

This will later support retrieval filtering.'''
from typing import Any, Dict, Optional


SUPPORTED_METADATA_FIELDS = (
    "document_type",
    "industry",
    "client",
    "project",
    "year",
    "department",
)


def normalize_metadata(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize document/chunk metadata to a predictable schema."""
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a mapping")
    result: Dict[str, Any] = {}
    for key in SUPPORTED_METADATA_FIELDS:
        value = metadata.get(key)
        if value is None or value == "":
            continue
        if key == "year":
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        elif isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        result[key] = value
    return result


def build_mongo_pre_filter(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the Atlas Vector Search pre-filter for metadata fields."""
    normalized = normalize_metadata(filters)
    # MongoDBAtlasVectorSearch stores LangChain Document metadata as
    # top-level document fields, not under a nested `metadata` object.
    return {key: value for key, value in normalized.items()}


def build_mongo_query(filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a normal MongoDB query for BM25/document filtering."""
    return build_mongo_pre_filter(filters)


def extract_metadata_from_document(
    filename: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = normalize_metadata(metadata)
    result["source"] = filename
    return result


def matches_metadata(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    normalized = normalize_metadata(filters)
    return all(metadata.get(key) == value for key, value in normalized.items())

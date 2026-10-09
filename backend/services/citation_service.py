'''Concept applied

Source Tracking / Page-Level Citations'''

'''adapt from : retrieval.py + metadata'''

from typing import Any, Dict, Iterable, List


def build_citations(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen = set()
    for item in results:
        metadata = item.get("metadata", {})
        key = (metadata.get("source"), metadata.get("page"), metadata.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "document": metadata.get("source", "unknown"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "chunk_type": metadata.get("chunk_type"),
            "document_type": metadata.get("document_type"),
            "industry": metadata.get("industry"),
            "year": metadata.get("year"),
            "score": item.get("rerank_score", item.get("fusion_score")),
        })
    return citations

'''Concept applied

Multi-retriever search

Result

For one question you can see:

Vector Results:
1. chunk_24
2. chunk_87
3. chunk_42

BM25 Results:
1. chunk_42
2. chunk_91
3. chunk_24

Now we have the foundation for hybrid retrieval.'''


'''also add Concept applied

Parent/Page Context Retrieval'''

'''This is particularly useful because a small chunk may not contain enough context.

For example:

Retrieved chunk
     ↓
Page 15
     ↓
Retrieve surrounding/page context
     ↓
Expanded context

Instead of giving the LLM only:

"Revenue increased by 18%."

you can provide the relevant surrounding content from the page.'''
'''Result

More complete context for the LLM while preserving the original retrieval granularity.'''

from typing import Any, Dict, List, Optional

from .metadata_service import build_mongo_pre_filter, build_mongo_query


class HybridRetriever:
    def __init__(self, vector_ingestor, collection=None):
        self.vector_ingestor = vector_ingestor
        # PyMongo Collection deliberately does not support truth-value
        # testing. Use an explicit None check instead of `collection or ...`.
        self.collection = (
            collection if collection is not None else vector_ingestor.collection
        )

    def vector_search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        docs = self.vector_ingestor.similarity_search_with_score(query, k=k, filters=filters)
        results = []
        for rank, (doc, score) in enumerate(docs, start=1):
            metadata = dict(doc.metadata or {})
            results.append({
                "text": doc.page_content,
                "metadata": metadata,
                "chunk_id": metadata.get("chunk_id"),
                "distance": float(score),
                "rank": rank,
                "retriever": "vector",
            })
        return results

    def bm25_search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        mongo_filter = build_mongo_query(filters)
        cursor = self.collection.find(mongo_filter, {"text": 1, "source": 1, "page": 1, "chunk_id": 1, "chunk_type": 1, "document_id": 1})
        rows = list(cursor)
        if not rows:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            from backend.services.bm25_service import BM25Okapi

        tokenized = [str(row.get("text", "")).lower().split() for row in rows]
        index = BM25Okapi(tokenized)
        scores = index.get_scores(query.lower().split())
        ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)[:k]

        results = []
        for rank, (row, score) in enumerate(ranked, start=1):
            metadata = {
                key: row[key]
                for key in ("source", "page", "chunk_id", "chunk_type", "document_id")
                if key in row
            }
            results.append({
                "text": row.get("text", ""),
                "metadata": metadata,
                "chunk_id": metadata.get("chunk_id"),
                "bm25_score": float(score),
                "rank": rank,
                "retriever": "bm25",
            })
        return results

    def hybrid_search(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None):
        return self.vector_search(query, k=k, filters=filters), self.bm25_search(query, k=k, filters=filters)


def expand_page_context(collection, results: List[Dict[str, Any]], radius: int = 1, max_chars: int = 6000) -> List[Dict[str, Any]]:
    """Expand each hit with nearby chunks from the same source/page."""
    expanded = []
    for result in results:
        metadata = result.get("metadata", {})
        source = metadata.get("source")
        page = metadata.get("page")
        if source is None or page is None:
            result["context"] = result.get("text", "")
            expanded.append(result)
            continue

        try:
            page_number = int(page)
        except (TypeError, ValueError):
            result["context"] = result.get("text", "")
            expanded.append(result)
            continue

        candidates = list(collection.find(
            {"source": source, "page": {"$gte": page_number - max(0, radius), "$lte": page_number + max(0, radius)}},
            {"text": 1, "source": 1, "page": 1, "chunk_id": 1},
        ).sort("page", 1))
        candidates.sort(key=lambda row: (row.get("page", 0), row.get("chunk_id", "")))
        context = "\n\n".join(row.get("text", "") for row in candidates if row.get("text"))
        result["context"] = context[:max_chars]
        expanded.append(result)
    return expanded

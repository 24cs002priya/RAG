'''Concept applied

Parent Page Retrieval / Context Expansion

Result

The LLM gets more context without requiring enormous chunks during indexing.'''

'''adapt parent-page retrieval idea from:

src/retrieval.py'''

'''Best Chunk
    ↓
Identify page
    ↓
Retrieve surrounding/full page
    ↓
Context for LLM'''

from typing import Any, Dict, List

from .retrieval import expand_page_context


class ContextService:
    def __init__(self, collection):
        self.collection = collection

    def expand(self, results: List[Dict[str, Any]], radius: int = 1, max_chars: int = 6000):
        return expand_page_context(self.collection, results, radius=radius, max_chars=max_chars)

    @staticmethod
    def build_context(results: List[Dict[str, Any]], max_chars: int = 18000) -> str:
        parts = []
        total = 0
        for item in results:
            metadata = item.get("metadata", {})
            header = f"[Source: {metadata.get('source', 'unknown')}, Page: {metadata.get('page', '?')}, Chunk: {metadata.get('chunk_id', '?')}]"
            text = item.get("context") or item.get("text", "")
            part = f"{header}\n{text}"
            if total + len(part) > max_chars:
                break
            parts.append(part)
            total += len(part)
        return "\n\n---\n\n".join(parts)

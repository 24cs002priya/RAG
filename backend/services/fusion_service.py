#Reciprocal Rank Fusion (RRF)

'''Vector Results
       +
BM25 Results
       ↓
Score/Rank Fusion
       ↓
Combined Results'''

'''Concept applied

Hybrid Search / Result Fusion

Result

Now your project genuinely implements:

Semantic Search
+
Keyword Search
↓
Hybrid Results'''

'''Workflow
Vector Results
      ↓
      ├────→ Fusion → Combined Results
      ↑
BM25 Results
Concept applied

Hybrid Search / Result Fusion

Result

A single ranked candidate list combining semantic and keyword retrieval.'''

from typing import Any, Dict, Iterable, List


class ReciprocalRankFusion:
    """Merge ranked vector and BM25 result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    @staticmethod
    def _key(item: Dict[str, Any]) -> str:
        return str(
            item.get("chunk_id")
            or item.get("id")
            or item.get("metadata", {}).get("chunk_id")
            or item.get("metadata", {}).get("id")
            or item.get("text", "")[:120]
        )

    def fuse(self, vector_results: Iterable[Dict[str, Any]], bm25_results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        scores: Dict[str, float] = {}

        for rank, item in enumerate(vector_results, start=1):
            key = self._key(item)
            merged.setdefault(key, dict(item))
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank)
            merged[key]["vector_rank"] = rank

        for rank, item in enumerate(bm25_results, start=1):
            key = self._key(item)
            merged.setdefault(key, dict(item))
            # Prefer populated metadata/text from the first retriever, but fill gaps.
            for field, value in item.items():
                if merged[key].get(field) in (None, "", {}):
                    merged[key][field] = value
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + rank)
            merged[key]["bm25_rank"] = rank

        result = []
        for key, item in merged.items():
            item["fusion_score"] = scores[key]
            result.append(item)
        return sorted(result, key=lambda x: x["fusion_score"], reverse=True)


class FusionService(ReciprocalRankFusion):
    pass


def reciprocal_rank_fusion(vector_results, bm25_results, k: int = 60):
    return ReciprocalRankFusion(k=k).fuse(vector_results, bm25_results)

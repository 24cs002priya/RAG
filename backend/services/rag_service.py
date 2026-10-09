'''Concept applied

RAG Pipeline Orchestration

Result

All individual components work as one complete system.

'''

'''answer_question(query)

        ↓

Query Analysis

        ↓

Metadata Filter

        ↓

Vector Search
      +
BM25 Search

        ↓

Fusion

        ↓

Reranking

        ↓

Parent Page Context

        ↓

LLM

        ↓

Answer + Citations'''

'''take refernce from pipeline.py'''

'''Ingestion pipeline
PDF
 ↓
pdf_parsing.py
 ↓
parsed_reports_merging.py
 ↓
tables_serialization.py
 ↓
text_splitter.py
 ↓
Metadata
 ↓
ingestion.py
 ├── Vector Index
 └── BM25 Index'''
'''Query pipeline
User Query
 ↓
questions_processing.py
 ↓
Metadata filtering
 ↓
retrieval.py
 ├── Vector
 └── BM25
 ↓
fusion.py
 ↓
reranking.py
 ↓
Parent/Page Context
 ↓
LLM
 ↓
Answer + Citations'''


import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from backend.data.database import get_collection
from backend.services.citation_service import build_citations
from backend.services.context_service import ContextService
from backend.services.fusion_service import ReciprocalRankFusion
from backend.services.indexing_service import VectorDBIngestor
from backend.services.questions_processing import QueryProcessor
from backend.services.retrieval import HybridRetriever
from backend.services.reranking import Reranker


class RAGService:
    def __init__(self):
        load_dotenv()
        self.collection = get_collection()
        self.vector = VectorDBIngestor()
        self.retriever = HybridRetriever(self.vector, self.collection)
        self.fusion = ReciprocalRankFusion()
        self.reranker = Reranker()
        self.context = ContextService(self.collection)
        self.query_processor = QueryProcessor()

    def _answer_with_llm(self, query: str, context: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # A useful non-LLM fallback keeps the API testable without credentials.
            return "I found the following relevant source material:\n\n" + context[:6000]

        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system = (
            "You are an internal consulting-firm knowledge assistant. "
            "Answer only from the supplied context. If the context is insufficient, say so. "
            "Do not invent figures, clients, dates, or conclusions. Mention source pages when useful."
        )
        response = client.chat.completions.create(
            model=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}"},
            ],
        )
        return response.choices[0].message.content or "No answer generated."

    def answer_question(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 8) -> Dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        processed = self.query_processor.process(query, filters)
        vector_results, bm25_results = self.retriever.hybrid_search(
            processed["retrieval_query"], k=max(top_k * 2, 10), filters=processed["filters"]
        )
        fused = self.fusion.fuse(vector_results, bm25_results)
        reranked = self.reranker.rerank(processed["retrieval_query"], fused, top_n=top_k)
        expanded = self.context.expand(reranked, radius=1)
        context = self.context.build_context(expanded)
        answer = self._answer_with_llm(query, context)
        return {
            "query": query,
            "answer": answer,
            "sources": build_citations(expanded),
            "applied_filters": processed["filters"],
        }


def answer_question(query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 8) -> Dict[str, Any]:
    """Convenience entry point used by the API layer."""
    return RAGService().answer_question(query, filters=filters, top_k=top_k)

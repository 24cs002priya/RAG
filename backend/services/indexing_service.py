from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

from backend.data.database import get_collection
from backend.services.metadata_service import build_mongo_pre_filter


class VectorDBIngestor:
    """
    Handles:
    1. Generating embeddings using HuggingFace
    2. Storing document chunks and embeddings
       in MongoDB Atlas Vector Search
    """

    def __init__(self):
        self.embeddings = self._setup_embeddings()
        self.collection = get_collection()

    def _setup_embeddings(self):
        """
        Initialize the HuggingFace embedding model.
        """
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def create_vector_store(
        self,
        documents: List[Document]
    ):
        """
        Create/store vector embeddings for the given documents.

        Each Document should already contain:
        - page_content
        - source
        - page
        - chunk_id
        - other metadata
        """

        if not documents:
            raise ValueError("No documents provided for indexing.")

        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
        )

        vector_store.add_documents(documents)

        return vector_store

    def add_documents(
        self,
        documents: List[Document]
    ):
        """
        Add chunks to the MongoDB Atlas Vector Search index.
        """

        if not documents:
            return 0

        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
        )

        vector_store.add_documents(documents)

        return len(documents)

    def delete_document(self, filename: str):
        """
        Delete all vector chunks belonging to a document.
        """

        result = self.collection.delete_many(
            {
                "source": filename
            }
        )

        return result.deleted_count

    def _vector_store(self):
        return MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name="vector_index",
            text_key="text",
            embedding_key="embedding",
        )

    def similarity_search(self, query: str, k: int = 5, filters=None):
        """
        Perform semantic/vector similarity search.
        """

        return self._vector_store().similarity_search(
            query, k=k, pre_filter=build_mongo_pre_filter(filters) or None
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filters=None,
    ):
        """
        Perform vector search and return similarity scores.
        """

        return self._vector_store().similarity_search_with_score(
            query, k=k, pre_filter=build_mongo_pre_filter(filters) or None
        )

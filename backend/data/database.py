import os

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

client = None
db = None


def get_db():
    """
    Get the MongoDB database connection.

    Database:
        rag_database
    """

    global client, db

    mongo_uri = os.getenv("MONGODB_URI")

    if not mongo_uri:
        raise RuntimeError(
            "MONGODB_URI must be set in .env"
        )

    if client is None:
        client = MongoClient(mongo_uri)

    if db is None:
        db = client["rag_database"]

    return db


def get_collection():
    """
    Collection used for RAG chunks and vector embeddings.
    """
    return get_db()["documents"]


def get_metadata_collection():
    """
    Collection used for document-level metadata.
    """
    return get_db()["documents_metadata"]
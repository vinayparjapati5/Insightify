import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API Key from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("Pinecone API key is missing!")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

def create_pinecone_index(user_id: str = None, is_private: bool = False) -> dict:
    """
    - Public RAG: Uses a shared database.
    - Private RAG: Creates a unique namespace for each user.

    Args:
        user_id (str, optional): Unique identifier for the user.
        is_private (bool): If True, creates a user-specific namespace.

    Returns:
        dict: Contains the index and namespace.
    """
    index_name = "rag-database"  # Shared index for both public & private RAG
    namespace = "public" if not is_private else f"user_{user_id}"  # Public namespace OR per-user namespace

    # Validate user_id for private namespaces
    if is_private and not user_id:
        raise ValueError("user_id must be provided for private namespaces!")

    # List existing indexes
    existing_indexes = pc.list_indexes()

    # Check if index exists
    if index_name not in existing_indexes:
        logger.info(f"Creating index '{index_name}'...")
        try:
            pc.create_index(
                name=index_name,
                dimension=768,  # Ensure this matches your vector dimensionality
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info(f"Index '{index_name}' created successfully.")
        except pinecone.PineconeException as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info(f"Index '{index_name}' already exists. Using the existing index.")
            else:
                logger.error(f"Failed to create index: {e}")
                raise e  # Re-raise the error if it's something else

    # Get the index reference
    index = pc.Index(index_name)

    return {"index": index, "namespace": namespace}
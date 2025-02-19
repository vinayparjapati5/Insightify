import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from services.pinecone_init import create_pinecone_index

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

retrieved_vector = None

#load the embeddings 
def get_embeddings():
    """Loads Google Generative AI embeddings."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            api_key=GOOGLE_API_KEY
        )
        return embeddings
    except Exception as e:
        print(f"Error: {e}")

#load the vector store
def vector_store(final_texts, user_id=None, is_private=False): #pass the user_id and is_private from the database.py file
    """
    Stores extracted text in Pinecone for retrieval.

    Args:
        final_texts (list): List of documents to store.
        user_id (str, optional): User ID for private RAG.
        is_private (bool, optional): Whether to use private RAG.

    Returns:
        dict: Contains a success message and the retriever.
    """
    global retrieved_vector
    try:
        embedding = get_embeddings()
        # Get Pinecone index and namespace
        pinecone_info = create_pinecone_index(user_id, is_private)
        index = pinecone_info["index"]
        namespace = pinecone_info["namespace"]
        # Initialize Pinecone vector store
        vector_store = Pinecone(index, embedding, namespace=namespace)
        # Store documents
        vector_store.add_documents(final_texts)
        # Set retriever
        retrieved_vector = vector_store
        return {"message": "Vector store created successfully", "retrieved_vector": retrieved_vector}
    except Exception as e:
        print(f" Error: {e}")
    

#retrive the data from the vector store based on the query
def retriver_data(vector_store):
    """Retrieves data from the vector store based on a query."""
    if vector_store is None:
        print("vector store is not created")
        return None
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        return retriever
    except Exception as e:
        print(f"Error: {e}")











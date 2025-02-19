import os
from dotenv import load_dotenv
from services.youtube_transcript import extract_transcript
from services.pdf_transcript import extract_pdf_transcript
from services.vector_store import vector_store, retriver_data, get_embeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.utils import filter_complex_metadata
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from services.excel import preprocessing_func
warnings.filterwarnings("ignore")
import google.generativeai as genai

# Load environment variables
load_dotenv()

print("..... Loading Keys .....")
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("langchain_api_key")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

print("..... Set Environment Variables .....")
os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["langchain_api_key"] = langchain_api_key
os.environ["HUGGINGFACE_API_KEY"] = huggingface_api_key

genai.configure(api_key=google_api_key)

# Load the LLM model
Model = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b",
    api_key=groq_api_key,
    temperature=0.2,
    max_tokens=2000,  # Reduce this value
    streaming=True,
    max_retries=3,
)

# Gemini = genai.GenerativeModel(
#     "gemini-1.5-flash")
    

# Global variable to store retriever
retriever = None


from langchain.schema import Document

def final_texts(transcript, pdf_text, whatsapp_text):
    if not transcript and not pdf_text and not whatsapp_text:
        raise ValueError("All inputs cannot be None.")

    final_texts = []

    #Append YouTube transcript
    if transcript:
        final_texts.append(Document(page_content=transcript, metadata={"source": "youtube"}))

    #Append PDF text 
    if pdf_text:
        final_texts.append(Document(page_content=pdf_text, metadata={"source": "pdf"}))
    
    #Append WhatsApp text
    if whatsapp_text:
        final_texts.append(Document(page_content=whatsapp_text, metadata={"source": "whatsapp"}))
    


    complex_metadata = filter_complex_metadata(final_texts)

    return complex_metadata


def process_vector_store(youtube_data, pdf_data, whatsapp_data):
    """Processes embeddings, stores data in vector DB, and initializes retriever."""
    global retriever  # Use global variable
    # Ensure we have data before processing
    if not pdf_data and not youtube_data and not whatsapp_data:
        return {"error": "No data available for vector store processing."}
    # Extract final texts
    text_data = final_texts(youtube_data, pdf_data,whatsapp_data)
    # Ensure extracted data is not empty
    if not text_data:
        return {"error": "No text extracted from provided files."}
    # Generate embeddings and store in vector DB
    vector = vector_store(text_data)
    if vector is None:
        raise ValueError("Vector store creation failed. Check input data and embeddings.")
    # Initialize retriever
    retriever = retriver_data(vector)
    if retriever is None:
        raise ValueError("Retriever initialization failed. Ensure vector store is working.")
    return {"message": "Vector store and retriever initialized successfully."}

# Define retrieval chain
def retrieval_chain(query,retriever):
    """Retrieves relevant context from the vector DB and uses LLM for answering queries."""
    if retriever is None:
        raise ValueError("Retriever is not initialized.")

    chain = RetrievalQA.from_chain_type(
        llm=Model,
        retriever=retriever,
        chain_type="stuff",
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    )
    return chain.invoke(query)



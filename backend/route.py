import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from services.pdf_transcript import extract_pdf_transcript
from services.youtube_transcript import extract_transcript
from main import final_texts, retrieval_chain
from services.vector_store import get_embeddings, vector_store, retriver_data
from main import filter_complex_metadata
from langchain.schema import Document
from services.whatsapp import extract_whatsapp_chat
from langchain_pinecone import Pinecone
from services.pinecone_init import create_pinecone_index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.excel import preprocessing_func
import warnings
import uvicorn
warnings.filterwarnings("ignore")



app = FastAPI()


#chunk the text
def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
    """
    Splits text into smaller overlapping chunks for better processing.
    
    Args:
        text (str): The input text to be chunked
        chunk_size (int): Maximum size of each chunk in characters
        chunk_overlap (int): Number of characters to overlap between chunks         
        
    Returns:
        dict: Dictionary containing the text chunks and the number of chunks
    """
    try:
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        return chunks
        
    except Exception as e:
        print(f"Error chunking text: {e}")
        return {"error": f"Failed to chunk text: {e}"}




# Store extracted text globally
extracted_data = {"pdf_text": None, "transcript": None, "whatsapp_text": None, "excel_text": None}
retrieved_vector = None  # Global retriever



#fr pdf transcript download
@app.post("/pdf")
async def preprocess_pdf(file: UploadFile = File(...)):
    save_dir = "temp_files"
    os.makedirs(save_dir, exist_ok=True)
    file_location = os.path.join(save_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    pdf_text = extract_pdf_transcript(file_location)
    extracted_data["pdf_text"] = pdf_text  # Save extracted text globally
    return {"message": "PDF processed successfully", "text": pdf_text}




#fr youtube transcript download
@app.post("/youtube")
async def process_youtube(url: str = Form(...)):
    transcript = extract_transcript(url)
    extracted_data["transcript"] = transcript  # Save transcript globally
    return {"message": "YouTube transcript processed successfully", "text": transcript}




#for load embeddings and vector store
@app.post("/embedding_vector_store_whatsapp")
async def embedding_vector_store_whatsapp(file: UploadFile = File(...)):
    save_dir = "temp_files"
    os.makedirs(save_dir, exist_ok=True)
    file_location = os.path.join(save_dir, file.filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    whatsapp_text = extract_whatsapp_chat(file_location)
    extracted_data["whatsapp_text"] = whatsapp_text if whatsapp_text is not None else []
    # Ensure at least one source of text is available
    if not whatsapp_text:
        return {"error": "No WhatsApp text available for processing."}
    # Convert whatsapp_text to a string
    whatsapp_text_str = " ".join(whatsapp_text) if isinstance(whatsapp_text, list) else whatsapp_text
    return {"message": "WhatsApp text processed successfully", "text": whatsapp_text_str}
    

#for excel file
@app.post("/excel")
async def process_excel(file: UploadFile = File(...)):
    save_dir = "temp_files"
    os.makedirs(save_dir, exist_ok=True)
    file_location = os.path.join(save_dir, file.filename or "")
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    excel_text = preprocessing_func(file_location)
    extracted_data["excel_text"] = excel_text if excel_text is not None else []
    return {"message": "Excel processed successfully", "text": excel_text}



@app.post("/embedding_vector_store_final_text")
async def embedding_vector_store_final_text(user_id: str = None, is_private: bool = False):
    """
    Processes extracted text from PDF, YouTube transcript, or WhatsApp, 
    generates embeddings, and stores them in Pinecone.
    
    Args:
        user_id (str, optional): User ID for private RAG.
        is_private (bool, optional): Whether to use private RAG.

    Returns:
        dict: Success message or error details.
    """
    global retrieved_vector
    # Ensure at least one text source is available
    pdf_text = extracted_data.get("pdf_text")
    transcript = extracted_data.get("transcript")
    whatsapp_text = extracted_data.get("whatsapp_text")
    excel_text  = extracted_data.get("excel_text")

    # Process WhatsApp text if available
    if not any([pdf_text, transcript, whatsapp_text, excel_text]):
        return {"error": "No text available for processing. Upload a PDF or provide a YouTube URL."}
    # Convert lists to strings (handling None values safely)
    def text_to_string(text):
        if isinstance(text, list):
            return " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in text])
        return text or ""

    pdf_text = text_to_string(pdf_text)
    transcript = text_to_string(transcript)
    whatsapp_text = text_to_string(whatsapp_text)
    excel_text = text_to_string([excel_text])


    # Combine all texts into a single list of documents
    final_texts = [pdf_text, transcript, whatsapp_text, excel_text]
    final_texts = [text.strip() for text in final_texts if text.strip()]  # Remove empty strings
    # Chunk the combined texts
    chunked_texts = []
    for text in final_texts:
        chunks = chunk_text(text)
        if chunks:  
            chunked_texts.extend(chunks)
    
    final_texts = chunked_texts  # Replace original texts with chunked versions

    if not final_texts:
        return {"error": "Final texts are empty, cannot proceed with vector storage."}

    # Convert text into LangChain Document format
    documents = [Document(page_content=text, metadata={"source": "uploaded_data"}) for text in final_texts]

    # Get Pinecone index & namespace
    try:
        pinecone_info = create_pinecone_index(user_id, is_private)
        index = pinecone_info["index"]
        namespace = pinecone_info["namespace"]
    except Exception as e:
        return {"error": f"Failed to initialize Pinecone index: {e}"}

    try:
        # Initialize Pinecone vector store
        embedding_function = get_embeddings()
        vector_store = Pinecone(index, embedding_function, namespace=namespace)
        # Store documents in Pinecone
        vector_store.add_documents(documents)
        retrieved_vector = vector_store  # Set retriever

        return {"message": "Embeddings, vector store, and final text processed successfully"}
    except Exception as e:
        return {"error": f"Failed to create Pinecone vector store: {e}"}



@app.post("/retrieval_chat")
async def retrieval(query: str = Form(...)):
    if retrieved_vector is None:
        return {"error": "Retriever is not initialized. Run '/embedding_vector_store_final_text' first."}
    retriever = retrieved_vector.as_retriever(search_kwargs={"k": 1})
    result = retrieval_chain(query,retriever)  # Perform retrieval
    return {"message": "Retrieval processed successfully", "retriever": result}

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
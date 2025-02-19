#this file is used to chunk the text into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        return []




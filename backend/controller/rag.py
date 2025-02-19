import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
import warnings
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi


warnings.filterwarnings("ignore")


load_dotenv()


#load all the apis keys here
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
langchain_api_key = os.getenv("langchain_api_key")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


print("Keys are loaded")
 

#set the enviroment
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["langchain_api_key"] = langchain_api_key
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"


print("Enviroment is set")


Gemini = ChatGoogleGenerativeAI(
    model="gemini-vision",
    api_key=GOOGLE_API_KEY,
)

print("Gemini is set")

Groq = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama3-8b-8192",
)

print("Groq is set")

#load the pdf /scan pdf etc
import pymupdf4llm
from langchain.text_splitter import MarkdownTextSplitter

def load_document(doc_path):
    """
    Load a document and convert it to markdown format with images
    
    Args:
        doc_path (str): Path to the document file
        
    Returns:
        str: Markdown text with embedded images
    """
    try:
        # Get the MD text
        md_text_images = pymupdf4llm.to_markdown(
            doc=doc_path,
            page_chunks=True, 
            write_images=True,
            image_path="images",
            image_format="png",
            dpi=300
        )
        return md_text_images
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None

pdf_path = r"C:\Users\HP\Desktop\Project\artificizen\backend\129049121_1734976290923.pdf"
result = load_document(pdf_path)
print("Text is loaded")  # Debugging step to check actual structure


texts = [] 
if isinstance(result, list) and len(result) > 0:  

    for item in result:  
        if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):  
            texts.append(item["text"])  


#extract the transcript from the youtube video
def extract_transcript(youtube_url):
    try:
        video_id = youtube_url.split("=")[1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ""
        for i in transcript:
            transcript_text += i['text']
        return transcript_text
    except Exception as e:
        print(f"Error: {e}")

youtube_url = ""
transcript = extract_transcript(youtube_url)
print("Transcript is extracted")


final_texts = []
if texts:
    final_texts.extend(texts)
if transcript:
    final_texts.append(transcript)
print("Final texts are loaded")
print(type(final_texts))

#Embeddings using langchain
print("Embeddings are loading")
from langchain_google_genai import GoogleGenerativeAIEmbeddings
def embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        api_key=GOOGLE_API_KEY,
    )
    return embeddings
embedding = embeddings()
print(embeddings)
print("Till here embeddings are set")


#store the data into the vector store
print("Storing the data into the vector store")

from langchain.vectorstores import Chroma
def chroma_db(embedding, final_texts):
    try:
        
        vectorstore = Chroma.from_texts(final_texts, embedding)
        return vectorstore
    except Exception as e:
        print(f"Error creating Chroma index: {str(e)}")

database = chroma_db(embedding, final_texts)
print("Database is created")

#make retriver of the database
retriver = database.as_retriever(search_kwargs={"k": 2})

#define the retriveQa chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
#define the chains
chain = RetrievalQA.from_chain_type(
    llm=Groq,
    retriever=retriver,
    chain_type="stuff",
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
)

res = chain.invoke("gave me a short summary of final text??")
print()
print()
print()
print()
print()
print("Resullt from the chain")
print(res)



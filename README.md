# Insightify - A Multimodal Retrieval-Augmented Generation (RAG) System

## Overview
This project is a **Multimodal Retrieval-Augmented Generation (RAG) system** built using **FastAPI**. It processes multiple data sources such as **PDFs, YouTube videos, WhatsApp chats, and Excel files**, generates embeddings, and stores them in **Pinecone** for retrieval. The system allows users to query stored knowledge and get relevant responses.

## Features
✅ **PDF Text Extraction**  
✅ **YouTube Transcript Retrieval**  
✅ **WhatsApp Chat Processing**  
✅ **Excel File Preprocessing**  
✅ **Text Chunking for Efficient Retrieval**  
✅ **Embedding Generation using LangChain**  
✅ **Vector Storage in Pinecone**  
✅ **Retrieval-Based Chatbot**  

## Tech Stack
- **Backend**: FastAPI
- **Database**: Pinecone (for vector storage)
- **NLP Models**: LangChain, Sentence Transformers
- **File Processing**: pdfplumber, Whisper, Pandas
- **Deployment**: Uvicorn

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the FastAPI Server**
```bash
uvicorn app:app --host 0.0.0.0 --port 10000 --reload
```

## API Endpoints
### **1. PDF Upload & Processing**
```http
POST /pdf
```
**Description**: Uploads and extracts text from a PDF file.

### **2. YouTube Transcript Extraction**
```http
POST /youtube
```
**Description**: Extracts transcript from a given YouTube video URL.

### **3. WhatsApp Chat Processing**
```http
POST /embedding_vector_store_whatsapp
```
**Description**: Uploads and processes WhatsApp chat files.

### **4. Excel File Processing**
```http
POST /excel
```
**Description**: Uploads and preprocesses an Excel file.

### **5. Store Embeddings & Vector Data**
```http
POST /embedding_vector_store_final_text
```
**Description**: Generates embeddings from uploaded text and stores them in Pinecone.

### **6. Retrieval Chatbot Query**
```http
POST /retrieval_chat
```
**Description**: Queries stored embeddings and retrieves relevant text.

## File Structure
```
multimodal-rag/
│── services/
│   ├── pdf_transcript.py  # PDF text extraction
│   ├── youtube_transcript.py  # YouTube transcript extraction
│   ├── whatsapp.py  # WhatsApp chat processing
│   ├── excel.py  # Excel file processing
│   ├── vector_store.py  # Embedding & vector storage
│   ├── pinecone_init.py  # Pinecone setup
│
│── main.py  # Core logic for text processing & retrieval
│── app.py  # FastAPI server
│── requirements.txt  # Dependencies
│── README.md  # Documentation
```

## License
This project is licensed under the **MIT License**.

---
🚀 **Contributions are welcome!** If you have any suggestions or improvements, feel free to create a PR.


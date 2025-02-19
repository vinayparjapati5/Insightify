import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"  # Your FastAPI server URL

# Page title
st.title("🎯Insightify Buddy")

# Initialize session state
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = "Public RAG"  # Default mode
if "username" not in st.session_state:
    st.session_state.username = None  # Store username for Private RAG
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}  # Store uploaded file data

# Sidebar for navigation
st.sidebar.title("🔗 Navigation")
rag_mode = st.sidebar.radio("Choose RAG Mode", ["Public RAG", "Private RAG"])

# Update session state with the selected RAG mode
st.session_state.rag_mode = rag_mode

# Ask for username only if Private RAG is selected
if rag_mode == "Private RAG":
    username = st.sidebar.text_input("👤 Enter Your Username")
    st.session_state.username = username  # Save username in session state
else:
    st.session_state.username = None  # Clear username for Public RAG

# Function to upload files
def upload_files():
    st.header("📤 Upload Files")
    
    # File upload options
    st.subheader("📂 Upload Your Files")
    file_type = st.radio("Select file type", ["PDF", "WhatsApp Chat", "Excel", "YouTube URL"])
    
    if file_type == "PDF":
        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file is not None:
            files = {"file": pdf_file.getvalue()}
            if rag_mode == "Private RAG" and st.session_state.username:
                files["username"] = st.session_state.username  # Include username for Private RAG
            response = requests.post(f"{BACKEND_URL}/pdf", files=files)
            if response.status_code == 200:
                st.success("✅ PDF processed successfully!")
                result = response.json()
                st.session_state.uploaded_files["pdf"] = result  # Save PDF data in session state
                st.write("📝 Extracted Text and Metadata:")
                st.text_area("Extracted Text", value=result.get("text", ""), height=300)
            else:
                st.error("❌ Failed to process PDF.")

    elif file_type == "WhatsApp Chat":
        whatsapp_file = st.file_uploader("Upload a WhatsApp chat ZIP file", type=["zip"])
        if whatsapp_file is not None:
            files = {"file": whatsapp_file.getvalue()}
            if rag_mode == "Private RAG" and st.session_state.username:
                files["username"] = st.session_state.username  # Include username for Private RAG
            response = requests.post(f"{BACKEND_URL}/embedding_vector_store_whatsapp", files=files)
            if response.status_code == 200:
                st.success("✅ WhatsApp chat processed successfully!")
                result = response.json()
                st.session_state.uploaded_files["whatsapp"] = result  # Save WhatsApp data in session state
                st.write("📝 Extracted Text:")
                st.text_area("Extracted Text", value=result, height=300)
            else:
                st.error("❌ Failed to process WhatsApp chat.")

    elif file_type == "Excel":
        excel_file = st.file_uploader("Upload an Excel file", type=["xlsx", "csv"])
        if excel_file is not None:
            files = {"file": excel_file.getvalue()}
            if rag_mode == "Private RAG" and st.session_state.username:
                files["username"] = st.session_state.username  # Include username for Private RAG
            response = requests.post(f"{BACKEND_URL}/excel", files=files)
            if response.status_code == 200:
                st.success("✅ Excel file processed successfully!")
                result = response.json()
                st.session_state.uploaded_files["excel"] = result  # Save Excel data in session state
                st.write("📊 Extracted Data:")
                st.json(result)
            else:
                st.error("❌ Failed to process Excel file.")

    elif file_type == "YouTube URL":
        youtube_url = st.text_input("Enter YouTube URL")
        if youtube_url:
            payload = {"url": youtube_url}
            if rag_mode == "Private RAG" and st.session_state.username:
                payload["username"] = st.session_state.username  # Include username for Private RAG
            response = requests.post(f"{BACKEND_URL}/youtube", data=payload)
            if response.status_code == 200:
                st.success("✅ YouTube transcript processed successfully!")
                result = response.json()
                st.session_state.uploaded_files["youtube"] = result  # Save YouTube data in session state
                st.write("📝 Extracted Transcript:")
                st.text_area("Extracted Transcript", value=result.get("text", ""), height=300)
            else:
                st.error("❌ Failed to process YouTube transcript.")

# Function to create embeddings
def create_embeddings():
    st.header("🧠 Create Embeddings")
    
    # Check if files have been uploaded
    if not st.session_state.uploaded_files:
        st.warning("⚠️ Please upload files before creating embeddings.")
        return
    
    if st.button("🚀 Create Embeddings"):
        payload = {"is_private": rag_mode == "Private RAG"}
        if rag_mode == "Private RAG" and st.session_state.username:
            payload["username"] = st.session_state.username  # Include username for Private RAG
        
        response = requests.post(
            f"{BACKEND_URL}/embedding_vector_store_final_text",
            data=payload
        )
        if response.status_code == 200:
            st.success("✅ Embeddings created successfully!")
            st.json(response.json())
        else:
            st.error("❌ Failed to create embeddings.")

# Function to query the retriever
def query_retriever():
    st.header("🔍 Query Retriever")
    
    query = st.text_input("❓ Enter your query")
    if st.button("🚀 Submit Query"):
        try:
            payload = {"query": query}
            if rag_mode == "Private RAG" and st.session_state.username:
                payload["username"] = st.session_state.username  # Include username for Private RAG
            
            response = requests.post(
                f"{BACKEND_URL}/retrieval_chat",
                data=payload  # Use `data` to send form data
            )
            if response.status_code == 200:
                st.success("✅ Query processed successfully!")
                result = response.json()

                # Extract the "result" field from the JSON response
                if "retriever" in result and "result" in result["retriever"]:
                    retrieved_result = result["retriever"]["result"]
                    st.write("📄 Retrieved Result:")
                    st.write(retrieved_result)  # Display only the "result" field
                else:
                    st.error("❌ The 'result' field is missing in the response.")
            else:
                st.error(f"❌ Failed to process query. Status code: {response.status_code}")
                st.write("Error details:", response.text)
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Main content based on selected RAG mode
if rag_mode == "Public RAG":
    st.sidebar.write("You are in **Public RAG** mode.")
    st.header("🌐 Public RAG")
    tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🧠 Create Embeddings", "🔍 Query Retriever"])
    
    with tab1:
        upload_files()
    with tab2:
        create_embeddings()
    with tab3:
        query_retriever()

elif rag_mode == "Private RAG":
    if st.session_state.username:
        st.sidebar.write(f"You are in **Private RAG** mode. Username: **{st.session_state.username}**")
        st.header("🔒 Private RAG")
        tab1, tab2, tab3 = st.tabs(["📤 Upload Files", "🧠 Create Embeddings", "🔍 Query Retriever"])
        
        with tab1:
            upload_files()
        with tab2:
            create_embeddings()
        with tab3:
            query_retriever()
    else:
        st.warning("⚠️ Please enter your username in the sidebar to use Private RAG.")
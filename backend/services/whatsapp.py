import os
import zipfile
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_whatsapp_chat(zip_path):
    extracted_folder = "unzipped_chats"
    os.makedirs(extracted_folder, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    chat_files = [f for f in os.listdir(extracted_folder) if f.endswith(".txt")]
    if not chat_files:
        print("No chat file found!")
        return None

    chat_file_path = os.path.join(extracted_folder, chat_files[0])

    # Read WhatsApp chat file
    try:
        with open(chat_file_path, "r", encoding="utf-8") as file:
            chat_text = file.readlines()

        # Apply Markdown formatting
        formatted_content = []
        for line in chat_text:
            line = line.strip()
            if line.startswith("[") and "]" in line:  
                formatted_content.append(f"**{line}**")  
            elif ":" in line:  
                formatted_content.append(f"**{line.split(':', 1)[0]}**: {line.split(':', 1)[1]}")  
            else:
                formatted_content.append(f"- {line}")  
        # Save the formatted content back to the original file
        with open(chat_file_path, "w", encoding="utf-8") as file:
            file.write("\n".join(formatted_content))

        print(f"Chat formatted and saved: {chat_file_path}")

        return [Document(page_content="\n".join(formatted_content))]

    except UnicodeDecodeError:
        print("Error decoding file. Try opening it with another encoding.")
        return None

    




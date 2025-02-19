import pymupdf4llm
def extract_pdf_transcript(doc_path):
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
            dpi=300,
        )
        # Extract text and metadata from markdown
        text_metadata = []
        for page in md_text_images:
            # Create dictionary for each page
            page_dict = {
                "text": page.get("text", ""),
                "metadata": {
                    "page_num": page.get("page"),
                    "source": "pdf"
                }
            }
            text_metadata.append(page_dict)    
        return text_metadata
    except Exception as e:
        print(f"Error loading document: {str(e)}")
        return None

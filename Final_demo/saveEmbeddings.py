import json
import tempfile
import threading
from processingTxt import split_chunks  # Adjust the import according to your module
from embeddings import get_embeddings  # Adjust the import according to your module
import os

def get_embeddings_for_chunks(chunks):
    """
    Generate embeddings for a list of chunks.
    
    Args:
    - chunks (list): List of chunks where each chunk contains page content and metadata.
    
    Returns:
    - list: List of dictionaries, each containing chunk ID, content, and embeddings.
    """
    embeddings = []
    for chunk in chunks:
        try:
            embedding = get_embeddings(chunk.page_content)
            embeddings.append({
                'chunk_id': chunk.metadata.get('id', 'unknown_id'),
                'chunk': chunk.page_content,
                'embedding': embedding.tolist()
            })
        except Exception as e:
            print(f"Error while generating embedding for chunk: {e}")
            embeddings.append({
                'chunk_id': chunk.metadata.get('id', 'unknown_id'),
                'chunk': chunk.page_content,
                'embedding': None
            })
    return embeddings

def process_pdf(temp_pdf_path, original_pdf_file_name):
    """
    Process the PDF to generate and return embeddings.
    
    Args:
    - temp_pdf_path (str): Path to the temporary PDF file.
    - original_pdf_file_name (str): Original name of the PDF file.
    
    Returns:
    - tuple: (list of messages, list of embeddings)
    """
    messages = []

    try:
        chunks = split_chunks(temp_pdf_path)
        messages.append(f"Type of chunks: {type(chunks)}")
        messages.append(f"Number of chunks: {len(chunks)}")
        if chunks:
            messages.append(f"First chunk ID: {chunks[0].metadata.get('id', 'unknown_id')}")

        embeddings = get_embeddings_for_chunks(chunks)
        messages.append("Embeddings generated successfully.")

    except Exception as e:
        messages.append(f"An error occurred during PDF processing: {e}")
        return messages, None

    return messages, embeddings

def generate_embeddings(uploaded_file):
    """
    Process the uploaded file to generate and return embeddings data.
    
    Args:
    - uploaded_file (UploadedFile): The uploaded file object from Streamlit.

    Returns:
    - tuple: (status message, list of embeddings)
    """
    if not uploaded_file:
        raise ValueError("Error: Please provide a valid PDF file.")
    
    def process_and_update():
        try:
            print("Processing the PDF and generating embeddings...")
            original_pdf_file_name = uploaded_file.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            messages, embeddings = process_pdf(temp_file_path, original_pdf_file_name)

            for message in messages:
                print(message)  # Display processing messages
            
            return "PDF content has been processed and embeddings are ready.", embeddings

        except Exception as e:
            print(f"Error: {e}")
            return f"An error occurred: {e}", None

        finally:
            os.remove(temp_file_path)

    # Start the processing in a separate thread
    status_message, embeddings = threading.Thread(target=process_and_update, daemon=True).start()
    return status_message, embeddings

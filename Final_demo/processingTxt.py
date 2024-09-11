import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
import numpy as np
from embeddings import get_embeddings

# Load documents from the PDF file
def load_documents(pdf_file_path):
    documents = []
    with pdfplumber.open(pdf_file_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            text = page.extract_text() or ''
            documents.append({
                'filename': os.path.basename(pdf_file_path),
                'text': text,
                'page': page_number  # Add page number to metadata
            })
    return documents

# Convert extracted documents to Document objects
def convert_to_documents(extracted_documents):
    documents = []
    for doc in extracted_documents:
        documents.append(Document(
            page_content=doc['text'],
            metadata={
                'filename': doc['filename'],
                'page': doc['page']  # Include page number in metadata
            }
        ))
    return documents

# Split the documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=80, 
        length_function=len, 
        is_separator_regex=False
    )
    # Split the documents and carry over metadata to each chunk
    chunks = []
    for document in documents:
        chunk_id = 0
        for chunk in text_splitter.split_documents([document]):
            chunk.metadata['id'] = f"{document.metadata['filename']}.{document.metadata['page']}.{chunk_id}"
            chunks.append(chunk)
            chunk_id += 1
    return chunks

# Main function to handle the PDF processing
def split_chunks(pdf_path):
    extracted_documents = load_documents(pdf_path)
    document_objects = convert_to_documents(extracted_documents)
    return split_documents(document_objects)

# Function to get embeddings for each chunk
def get_embeddings_for_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embeddings(chunk.page_content)
        embeddings.append({
            'chunk': chunk.page_content,
            'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
        })
    return embeddings

# Function to save embeddings to a file
def save_embeddings_to_file(embeddings, file_path):
    with open(file_path, 'w') as file:
        json.dump(embeddings, file, indent=4)
    print(f"Embeddings saved to {file_path}")

def main():
    print("Starting the process...")
    
    # Define the directory for storing embeddings
    embeddings_dir = 'embeddings'
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    
    # PDF file path (for demonstration purposes; you can modify this to be dynamic)
    pdf_path = "machine.pdf"
    
    # Generate a file name based on the PDF path or document ID
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]  # Use the file name without extension as document ID
    output_file_path = os.path.join(embeddings_dir, f'{document_id}_embeddings.json')

    # Check if the embeddings file already exists
    if os.path.exists(output_file_path):
        print(f"Embeddings for {document_id} already exist at {output_file_path}.")
    else:
        try:
            chunks = split_chunks(pdf_path)  # Get chunks from the PDF
            embeddings = get_embeddings_for_chunks(chunks)  # Get embeddings for chunks
            save_embeddings_to_file(embeddings, output_file_path)  # Save embeddings to a file
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

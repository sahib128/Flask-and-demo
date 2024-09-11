import json
import tempfile
import streamlit as st
from processingTxt import split_chunks  # Adjust the import according to your module
from embeddings import get_embeddings  # Adjust the import according to your module
from chatbot import query_rag, query_general_model
import os

def get_embeddings_for_chunks(chunks):
    """
    Generate embeddings for a list of chunks.

    Args:
    - chunks (list): List of chunks where each chunk contains page content and metadata.

    Returns:
    - str: JSON string of dictionaries, each containing chunk ID, content, and embeddings.
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
    return json.dumps(embeddings)

def process_pdf(temp_pdf_path):
    """
    Process the PDF to generate and return embeddings.

    Args:
    - temp_pdf_path (str): Path to the temporary PDF file.

    Returns:
    - tuple: (list of messages, JSON string of embeddings)
    """
    messages = []
    try:
        chunks = split_chunks(temp_pdf_path)
        messages.append(f"Type of chunks: {type(chunks)}")
        messages.append(f"Number of chunks: {len(chunks)}")
        if chunks:
            messages.append(f"First chunk ID: {chunks[0].metadata.get('id', 'unknown_id')}")

        embeddings_json = get_embeddings_for_chunks(chunks)
        messages.append("Embeddings generated successfully.")

    except Exception as e:
        messages.append(f"An error occurred during PDF processing: {e}")
        return messages, None

    return messages, embeddings_json

def generate_embeddings(uploaded_file):
    """
    Process the uploaded file to generate and return embeddings data.

    Args:
    - uploaded_file (UploadedFile): The uploaded file object from Streamlit.

    Returns:
    - tuple: (status message, JSON string of embeddings)
    """
    if not uploaded_file:
        raise ValueError("Error: Please provide a valid PDF file.")
    
    try:
        print("Processing the PDF and generating embeddings...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        messages, embeddings_json = process_pdf(temp_file_path)
        os.remove(temp_file_path)  # Clean up temporary file

        for message in messages:
            print(message)  # Display processing messages
        
        return "PDF content has been processed and embeddings are ready.", embeddings_json

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {e}", None

def handle_query():
    """
    Handle user queries using the selected model and embeddings directly.
    """
    try:
        model = selected_model if selected_model else "llama3.1"
        
        if st.session_state.embeddings_data:
            response = query_rag(st.session_state.query_input, st.session_state.embeddings_data, model)
        else:
            response = query_general_model(st.session_state.query_input, model)
        
        st.session_state.messages.append({"role": "user", "content": st.session_state.query_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.query_input = ""  # Clear input field
        return response

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")
    
    # Initialize session state attributes if they do not exist
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    if "embeddings_data" not in st.session_state:
        st.session_state.embeddings_data = None
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None

    with st.sidebar:
        st.title('📚💬 CHAT')

        st.subheader('Models and Parameters')
        model_options = ["llama3.1", "mistral", "another_model_2"]
        global selected_model
        selected_model = st.sidebar.selectbox("Select Model", model_options, index=model_options.index("llama3.1"))
        temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.sidebar.slider('Max Length', min_value=32, max_value=128, value=120, step=8)
        
        st.sidebar.header("Upload PDF")
        uploaded_file = st.sidebar.file_uploader("", type="pdf")
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file is None or st.session_state.uploaded_file.name != uploaded_file.name:
                # Process the uploaded PDF and generate embeddings
                status_message, embeddings_json = generate_embeddings(uploaded_file)

                # Add status message to chat history
                st.session_state.messages.append({"role": "system", "content": status_message})
                
                # Update the embeddings data if processing was successful
                if "PDF content has been processed and embeddings are ready." in status_message:
                    st.session_state.embeddings_data = embeddings_json
                    st.session_state.uploaded_file = uploaded_file

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
        st.button('Clear Chat History', on_click=clear_chat_history)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.query_input = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Handle the query and get the response
        response = handle_query()
        
        with st.chat_message("assistant"):
            st.write(response)

# Run the main function
if __name__ == '__main__':
    main()

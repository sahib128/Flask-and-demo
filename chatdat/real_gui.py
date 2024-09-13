import json
import tempfile
import streamlit as st
from saveEmbeddings import generate_embeddings  # Adjust the import according to your module
from embeddings import get_embeddings  # Adjust the import according to your module
from chatbot import query_rag, query_general_model
import os

def handle_query(temperature: float, top_p: float, max_length: int):
    """
    Handle user queries using the selected model and embeddings directly.
    """
    try:
        model = selected_model if selected_model else "llama3.1"
     
        response_placeholder = st.empty()
        
        # Use the general model if no embeddings are available
        if st.session_state.embeddings_data:
            response = query_rag(
                st.session_state.query_input, 
                st.session_state.embeddings_data, 
                model,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length
            )
        else:
            response = query_general_model(
                st.session_state.query_input, 
                model,
                temperature=temperature,
                top_p=top_p,
                max_length=max_length
            )
        
        full_response = ""
        for chunk in response:
            full_response += chunk
            response_placeholder.write(full_response)
        
        # Update session state with the responses
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.query_input = ""  # Clear input field

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
        
        temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01, key='temperature_slider')
        top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01, key='top_p_slider')
        max_length = st.sidebar.slider('Max Length', min_value=32, max_value=128, value=120, step=8, key='max_length_slider')
        
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

        # Handle file cancellation or removal
        if uploaded_file is None and st.session_state.uploaded_file is not None:
            st.session_state.embeddings_data = None
            st.session_state.uploaded_file = None
            st.session_state.messages.append({"role": "system", "content": "PDF content has been removed. The chatbot will now function as a general model."})
        
        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        
        st.button('Clear Chat History', on_click=clear_chat_history)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle new input
    if prompt := st.chat_input():
        st.session_state.query_input = prompt
        
        # Add the user input to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Handle the query and get the response
        handle_query(temperature, top_p, max_length)

# Run the main function
if __name__ == '__main__':
    main()

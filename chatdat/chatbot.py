from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import numpy as np
import json

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def load_model(model_name: str):
    """
    Load and return the model based on the model_name.
    
    Args:
    - model_name (str): The name of the model to load.
    
    Returns:
    - model: The loaded model.
    """
    return Ollama(model=model_name)  # Instantiate the Ollama model with the chosen model_name

def handle_prompt(query_text: str, context_text: str, model, temperature: float, top_p: float, max_length: int):
    """
    Handle the prompt and get a response from the model.
    
    Args:
    - query_text (str): The query from the user.
    - context_text (str): The context to be used for answering the query.
    - model: The model used to generate the response.
    
    Returns:
    - response_text (str): The response generated by the model.
    """
    # Create prompt for the model
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get the response from the model
    for chunk in model.stream(prompt, temperature=temperature, top_p=top_p, max_length=max_length):
         yield chunk


def query_rag(query_text: str, embeddings_json: str, model_name: str, temperature: float, top_p: float, max_length: int):
    """
    Query the model with context from JSON embeddings.
    
    Args:
    - query_text (str): The query from the user.
    - embeddings_json (str): JSON string of embeddings.
    - model_name (str): The name of the model to load.
    
    Returns:
    - response (str): The response generated by the model.
    """
    # Parse the JSON string to get embeddings
    embeddings = json.loads(embeddings_json)

    # Use the first chunk's data as context
    if embeddings:
        embeddings_dict = embeddings[0]
        context_text = embeddings_dict.get('chunk', '')
    else:
        context_text = ""
    
    # Load the model
    model = load_model(model_name)
    
    # Get the response from the model with context
    for response_chunk in handle_prompt(query_text, context_text, model, temperature, top_p, max_length):
        yield response_chunk


def query_general_model(query_text: str, model_name: str, temperature: float, top_p: float, max_length: int):
    """
    Query the general model without specific context.
    
    Args:
    - query_text (str): The query from the user.
    - model_name (str): The name of the model to load.
    
    Returns:
    - response (str): The response generated by the model.
    """
    # General model context is empty
    context_text = ""
    
    # Load the model
    model = load_model(model_name)
    
    # Handle the prompt with the loaded model
    for chunk in model.stream(query_text, temperature=temperature, top_p=top_p, max_length=max_length):
         yield chunk

# Example usage:
# response = query_rag("What is the context of this document?", '{"chunk": "Example context text", "embedding": [0.1, 0.2, 0.3]}', "llama3.1")
# print(response)

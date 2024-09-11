from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, pipeline
import torch
import os
from flask import Flask, request, jsonify

# Set the Hugging Face token (if needed for authentication)
os.environ['HF_TOKEN'] = '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'

# Define your model name (replace with your GGUF model name)
MODEL_NAME = 'arcee-ai/Llama-3.1-SuperNova-Lite'

# Load the model configuration
def load_model_config(model_name: str):
    # Load configuration
    config = AutoConfig.from_pretrained(model_name, use_auth_token=os.getenv('HF_TOKEN'))
    
    # Check and fix configuration issues if needed
    # For example, fixing rope_scaling if required
    if 'rope_scaling' in config.to_dict():
        config.rope_scaling = {
            'name': 'default',
            'factor': 8.0
        }

    return config

# Load the model and tokenizer
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv('HF_TOKEN'))
    config = load_model_config(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config, use_auth_token=os.getenv('HF_TOKEN'))
    return tokenizer, model

# Initialize the model and tokenizer
tokenizer, model = load_model(MODEL_NAME)

# Prompt template
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based only on the following context:

{context}

---

Question: {question}
"""

def handle_prompt(query_text: str, context_text: str, tokenizer, model):
    # Create the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    
    # Use the tokenizer and model to get a response
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Set max_new_tokens to control the length of the output
    response = pipe(prompt, max_new_tokens=150, num_return_sequences=1)[0]['generated_text']
    
    # Extract the answer from the response
    answer_start = response.find('Answer:')
    if answer_start != -1:
        response = response[answer_start + len('Answer:'):].strip()
    else:
        response = response.strip()
    
    return response

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query_text = data.get('query', '')
    context_text = data.get('context', '')

    if not query_text:
        return jsonify({'error': 'No query text provided'}), 400

    if not context_text:
        return jsonify({'error': 'No context provided'}), 400

    # Directly handle the query and context
    response = handle_prompt(query_text, context_text, tokenizer, model)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

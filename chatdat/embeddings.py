import warnings
from transformers import logging as transformers_logging

# Suppress specific warnings related to BERT parameter renaming
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`.")
warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`.")
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set. It will be set to `True` by default.")

# Set the logging level for transformers library to error only
transformers_logging.set_verbosity_error()

# Now, load the pre-trained BERT tokenizer and model
from transformers import BertModel, BertTokenizer
import torch
import numpy as np

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embeddings(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embeddings for the input tokens
    embeddings = outputs.last_hidden_state.squeeze(0)  # Shape: (sequence_length, hidden_size)

    # Convert the embeddings to numpy array
    embeddings_np = embeddings.cpu().numpy()

    # Compute the average embedding across all tokens
    avg_embedding = np.mean(embeddings_np, axis=0)
    print("Used BERT")

    return avg_embedding
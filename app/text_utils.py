import numpy as np
from numpy.linalg import norm
from typing import List, Tuple
from openai import OpenAI
import openai
import warnings
warnings.filterwarnings("ignore")

def cosine_similarity(vector_a, vector_b):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = norm(vector_a)
    norm_b = norm(vector_b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def create_embeddings(txt_list: List[str], model='text-embedding-3-small') -> List[np.ndarray]:
    """
    Generates embedding vectors for a given list of texts.

    Args:
        txt_list (List[str]): A list of texts to generate embeddings for.
        model (str, optional): The embedding model to use.

    Returns:
        List[np.ndarray]: Each embedding vector.
    """

    client = OpenAI()

    response = client.embeddings.create(
    input=txt_list,
    model=model)
    responses = [r.embedding for r in response.data]

    return responses

def normal_chat_completion(input_prompt: str, model: str = 'gpt-4-turbo-preview') -> dict:
    """
    Generates a JSON output using Openai chat completion.

    Args:
        input_prompt (str): The input prompt to the chat model.
        model (str, optional): Model name. Defaults to 'gpt-4-turbo-preview'.

    Returns:
        dict: The chat completion response formatted as a JSON object.
    """
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": 'You are a smart and intelligent program that understands information and provides output in JSON format'},
            {"role": "user", "content":input_prompt}
        ]
        )
    return response

def search_similar_vector(query_feature: np.array, features: List[np.array], topk: int = 10) -> Tuple[np.array, np.array]:
    """
    Compares with the given vectors and provides the index and similarity of the vector similar to query_feature.

    Args:
        query_feature (np.array): input embedding vector
        features (List[np.array]): list of embedding vectors
        topk (int, optional): number of top-k similar vectors

    Returns:
        Tuple[np.array, np.array]: index & cosine similarity of similar embedding vectors
    """
    features_stack = np.vstack(features)
    
    similarities = cosine_similarity([query_feature], features_stack).flatten()
    sorted_indices_desc = np.argsort(similarities)[::-1]
    
    # Get top-k indices
    topk_indices = sorted_indices_desc[:topk]
    # Retrieve the top-k cosine similarities
    topk_similarities = similarities[topk_indices]

    return topk_indices, topk_similarities
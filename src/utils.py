from owlready2 import *
import re
from sentence_transformers import SentenceTransformer
import numpy as np


def create_embedding(text:str) -> np.array:
    """creates the embedding using the sentence transformers module

    Args:
        text (str): the text to be embedded

    Returns:
        np.array: the embedding array
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding
    
    
def preprocess_pascalCamel(text: str) -> str:
    """decomposes a pascalCase or CamelCase string into a a more natural representation of the language

    Args:
        pascal (str): _description_

    Returns:
        str: _description_
    """
    list = re.split(r'(?=[A-Z])', text)
    list = [l.lower() for l in list]
    result = " ".join(l for l in list)
    result = result.strip()
    return result

def format_chroma_results(results) -> dict:
    formatted_results = []
    for i in range(len(results["ids"][0])):  
        formatted_results.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return formatted_results
    
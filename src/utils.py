
from owlready2 import *
import re
from sentence_transformers import SentenceTransformer



def create_embedding(text:str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding
    
    
def preprocess_pascalCase(pascal: str) -> str:
    list = re.split(r'(?=[A-Z])', pascal)
    list = [l.lower() for l in list]
    result = " ".join(l for l in list)
    return result

def format_chroma_results(results):
    formatted_results = []
    for i in range(len(results["ids"][0])):  
        formatted_results.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return formatted_results
    
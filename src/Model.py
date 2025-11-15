import torch
import torch.nn
from chromadb import Documents, EmbeddingFunction, Embeddings
from Ontology import Ontology

class ClassTopology(EmbeddingFunction):
    
    def __init__(self, ontology : Ontology):
        self.ontology = ontology
    
    def __call__(self, docs : Documents) -> Embeddings:
        pass
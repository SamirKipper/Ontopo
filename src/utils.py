from owlready2 import *
import re
import owlready2.entity
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.corpus import wordnet
from owlready2 import *

##* NLP CONCEPTS
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
    
##? DO I NEED THIS WITH NEO4J?
def create_embedding_dict(ontology):
    label_dict = dict()
    def_dict = dict()
    for c in ontology.classes():
        ## ! still needs correct error handling
        try:
            label_vec = create_embedding(c.label[0])
            label_dict[c.iri] = label_vec
        except:
            print("no label found")
            
        try:
            def_vec = create_embedding(c.definition[0])
            def_dict[c.iri] = def_vec
        except:
            ### ! create proper function that handles this properly
            set = wordnet.synsets(c.label[0])
            item = set[0]
            defi = item.definition()
            def_vec = create_embedding(defi)
            def_dict[c.iri] = def_vec
            print(f"No definition for {c.iri}, gave definition : {defi}")



##* ONTOLOGY CONCEPTS

def disect_AND_construction(intersection:And) -> list:
    return [c for c in intersection.Classes]

def get_negated(statement:Not):
    return statement.Class



if __name__ == "__main__":
    file = "/root/github/EmbedAlign/test/bfo-core.owl"
    bfo = get_ontology(file).load()
    bfo.get_namespace("http://purl.obolibrary.org/obo/")
    bearer = bfo.search(label = "bearer of")[0]
    dom = bearer.domain[0]
    dissected = disect_AND_construction(dom)
    dissected = [d for d in dissected if type(d) == owlready2.entity.ThingClass]
    print(dissected)
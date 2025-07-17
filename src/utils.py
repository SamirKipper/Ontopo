
import re
import owlready2.entity

import numpy as np
from nltk.corpus import wordnet
from pyoxigraph import *
from OWL import Operators

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
    



##* OXI UTILS





def extract_rdf_list(store: Store, list_node) -> list:
        """Traverses an RDF list starting from list_node"""
        items = []
        current = list_node
        while current != Operators["RDF_NIL"]:
            first_quads = list(store.quads_for_pattern(current, Operators["RDF_FIRST"], None, None))
            if not first_quads:
                break
            items.append(first_quads[0].object)
            rest_quads = list(store.quads_for_pattern(current, Operators["RDF_REST"], None, None))
            if not rest_quads:
                break
            current = rest_quads[0].object
        return items


def pop_blank_to_class_list(bnode: BlankNode, store: Store) -> list:
    """
    Extracts named classes from a class expression in a blank node,
    excluding any class found inside owl:complementOf.
    
    Args:
        bnode (BlankNode): The root blank node of the class expression.
        store (Store): The PyOxigraph store.

    Returns:
        list: A list of NamedNode URIs representing included classes.
    """
    results = []
    visited = set()
    queue = [bnode]

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)

        # Skip this node entirely if it's an owl:complementOf wrapper
        comp_quad = next(store.quads_for_pattern(current, Operators["OWL_COMPLEMENT_OF"], None, None), None)
        if comp_quad:
            continue  # skip complements entirely

        for operator in Operators.keys():
            op_quad = next(store.quads_for_pattern(current, Operators[operator], None, None), None)
            if op_quad:
                list_node = op_quad.object
                elements = extract_rdf_list(store, list_node)
                for el in elements:
                    if isinstance(el, NamedNode):
                        results.append(el)
                    elif isinstance(el, BlankNode):
                        queue.append(el)
    return results
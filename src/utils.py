
import re
from pyoxigraph import *
import networkx as nx
from typing import Generator

from OWL import OWL
from RDF import RDF
from RDFS import RDFS

##* NLP UTILS
    
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
    
def format_chroma_results(results) -> list:
    """formats the chroma results to a dictionary for each result
    
    Args:
        results (_type_): results froma chroma query

    Returns:
        list of dictionaries: each dictionary in the list contains key value pairs of the query results
    """
    formatted_results = []
    for i in range(len(results["ids"][0])):  
        formatted_results.append({
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    return formatted_results





##* OXI UTILS

def extract_rdf_list( list_node, store: Store,) -> list:
        """Traverses an RDF list starting from list_node"""
        items = []
        current = list_node
        while current != RDF.nil:
            first_quads = list(store.quads_for_pattern(current, RDF.first, None, None))
            if not first_quads:
                break
            items.append(first_quads[0].object)
            rest_quads = list(store.quads_for_pattern(current, RDF.rest, None, None))
            if not rest_quads:
                break
            current = rest_quads[0].object
        return items

def get_subclasses(OntoClass : NamedNode, store:Store) -> Generator:
    return (c.subject for c in store.quads_for_pattern(None, RDFS.subClassOf, OntoClass, None) if isinstance(c,NamedNode))
    

def get_descendents(OntoClass:NamedNode, store:Store) -> list:
    subclasses = get_subclasses(OntoClass, store)
    results = []
    if subclasses:
        queue = subclasses
        while queue:
            current = queue.pop()
            results.append(current)
            q_subs = get_subclasses(current,store)
            if q_subs:
                queue.extend(q_subs)
    return results

def get_classes_list(store:Store) -> list:
    quads = store.quads_for_pattern(None, RDF.type, OWL.Class, None)
    classes = [q.subject for q in quads if type(q.subject) == NamedNode]
    return classes

def get_all_classes(store):
    return (q.subject for q in store.quads_for_pattern(None, RDF.type, OWL.Class, None))

def get_complement(OntoClass : NamedNode, store : Store) -> list:
    desc = get_descendents(OntoClass, store)
    class_group = desc + [OntoClass]
    classes = get_classes_list(store)
    complement = [item for item in classes if item not in class_group]
    return complement

def handle_union(blanknode, store):
    union_list = extract_rdf_list(blanknode, store)
    return union_list

def handle_intersection(blanknode, store):
    pass

def pop_blank(blanknode : BlankNode, store) -> list:
    blank_quads = list(store.quads_for_pattern(blanknode, None, None, None))
    popped = None
    op_type = None
    match blank_quads[0].predicate:
        case OWL.unionOf:                                   ## NOTE: stored as list
            union_blank = blank_quads[0].object
            popped = extract_rdf_list(union_blank, store)
            op_type = OWL.unionOf
        case OWL.intersectionOf:                            ## NOTE: stored as list 
            intersection_blank = blank_quads[0].object
            popped = extract_rdf_list(intersection_blank, store)
            op_type = OWL.intersectionOf
        case OWL.complementOf:                              ## NOTE: points to class
            popped = [blank_quads[0].object]
            op_type = OWL.complementOf
        case OWL.Restriction:
            return blank_quads, OWL.Restriction
        case _:
            return list(blank_quads), None
    return popped, op_type


def get_dom_ran(poppable: NamedNode | BlankNode, store : Store):
    if isinstance(poppable, NamedNode):
        return get_descendents(poppable) + [poppable]
    if isinstance(poppable, BlankNode):
        return pop_blank(poppable, store)


from pyoxigraph import BlankNode, Store, NamedNode, Quad
from typing import List, Generator
import re
from transformers import AutoTokenizer, AutoModel
import torch

from OWL import OWL
from RDF import RDF
from RDFS import RDFS






def preprocess_pascalCase(pascal: str) -> str:
    """preprocesses pascalcase writing to regular phrases

    Args:
        pascal (str): a string in pascalCase

    Returns:
        str: a regular phrase
    """
    list = re.split(r'(?=[A-Z])', pascal)
    list = [l.lower() for l in list]
    result = " ".join(l for l in list)
    return result

def format_chroma_results(results):
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


### RDF UTILS

def extract_rdf_list(list_node : BlankNode, store: Store) -> list:
    """traverses an RDF list and returns the list elements

    Args:
        list_node (BlankNode): the BlankNode, that represents the list
        store (Store): the store, in which the blank node is stored in

    Returns:
        list: _description_
    """
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


def pop_blank_to_dict(blanknode : BlankNode, store : Store) -> dict:
    quads = store.quads_for_pattern(blanknode, None, None, None)
    return_dict = dict()
    for q in quads:
        return_dict[q.predicate] = q.object
    return return_dict


def check_quads_for_op(quads : List[Quad]) -> List[NamedNode]:
    op_types = [
        OWL.unionOf, OWL.intersectionOf, OWL.complementOf,
        OWL.Restriction, OWL.onProperty, OWL.oneOf
    ]
    preds = [q.predicate for q in quads]
    in_side = [p for p in preds if p in op_types]
    if len(in_side) == 0:
        print(quads)
    return in_side


def pop_blank(blanknode : BlankNode, store : Store) -> tuple:
    """opens a blank node and looks into its contents. 
    Can only return values for union, intersection or complement blanks

    Args:
        blanknode (BlankNode): the blank node to be popped
        store (Store): the store reference for the blank node

    Returns:
        tuple: list of popped elements, type of operator found (in string)
        
    Example:
    Assume we have the following turtle
    ```
    @prefix ex: <http://example.org#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    
    ex:classA   rdf:type owl:Class;
                rdfs:subClassOf [
                                rdf:type owl:Class;
                                owl:unionOf (
                                            ex:classB
                                            ex:classc
                                            )
                                ]
    
    ex:classB rdf:type owl:Class.
    
    ex:classC rdf:type owl:Class.
    
    
    ```
    we use the function as follows
    
    
    ```python
    store = Store(path = "path/to/file")
    classes = get_classes(store)
    for c in classes:
        if isinstance(c, NamedNode):
            pass
        else:
            popped, type = pop_blank(c, store)
    ```
    the outout will be
    ```python
    popped = [classB, classC]
    
    
    ```
    """
    blank_quads = list(store.quads_for_pattern(blanknode, None, None, None))
    popped = None
    op_types = check_quads_for_op(blank_quads)
    if len(op_types) == 1:
        op_type = op_types[0]
        op_quad = list(store.quads_for_pattern(blanknode, op_type, None, None))
        try:
            match op_type:
                ##* union, intersection complement
                case OWL.unionOf:                                   
                    union_blank = op_quad[0].object
                    popped = extract_rdf_list(union_blank, store)
                case OWL.intersectionOf:                           
                    intersection_blank = op_quad[0].object
                    popped = extract_rdf_list(intersection_blank, store)
                case OWL.complementOf:                             
                    popped = [op_quad[0].object]
                ##* restrictions
                case OWL.Restriction:
                    popped = dict()
                    for q in blank_quads:
                        popped[q.predicate] = q.object
                case OWL.onProperty:
                    popped = dict()
                    for q in blank_quads:
                        popped[q.predicate] = q.object
                case OWL.oneOf:
                    pass
        except Exception as e:
            raise TypeError(f"operator type likely not handled: {op_type}, {str(e)}")
    elif len(op_types) == 0:
        raise ValueError(f"no operator type detected")
    else:
        raise ValueError("multiple operator types detected in one blank node")
    if popped:
        return popped, op_type
    else:
        raise ValueError(f"no list elements found for op type {op_type}")


def get_subclasses(OntoClass : NamedNode, store:Store) -> Generator:
    return (c.subject for c in store.quads_for_pattern(None, RDFS.subClassOf, OntoClass, None) if isinstance(c,NamedNode))


def get_dom_ran(poppable: NamedNode | BlankNode, store : Store):
    if isinstance(poppable, NamedNode):
        return get_descendents(poppable) + [poppable]
    if isinstance(poppable, BlankNode):
        return pop_blank(poppable, store)


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

def get_complement(OntoClass : NamedNode, store : Store) -> list:
    desc = get_descendents(OntoClass, store)
    class_group = desc + [OntoClass]
    classes = get_classes_list(store)
    complement = [item for item in classes if item not in class_group]
    return complement
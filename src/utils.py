import json
import numpy as np
from pyoxigraph import BlankNode, Store, NamedNode, Quad
from typing import List
import networkx as nx
from typing import Generator
from sentence_transformers import SentenceTransformer
import subprocess
import re
import matplotlib.pyplot as plt
from adjustText import adjust_text 

from OWL import OWL
from RDF import RDF
from RDFS import RDFS

repo_root = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()


def plot_hierarchy(graph):
    sinks = [n for n, d in graph.out_degree() if d == 0]
    if len(sinks) != 1:
        raise ValueError("Graph must have exactly one sink node")
    sink = sinks[0]

    G_rev = graph.reverse()
    pos = hierarchy_pos(G_rev, root=sink)

    # Scale positions to increase spacing
    scale_factor = 2  # tweak this as needed
    pos = {k: (v[0]*scale_factor, v[1]*scale_factor) for k,v in pos.items()}

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, arrows=True)

    # Draw labels separately and use adjustText to prevent overlaps
    texts = []
    for node, (x, y) in pos.items():
        texts.append(plt.text(x, y, str(node), fontsize=10, ha='center', va='center'))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

    plt.axis('off')
    plt.show()




def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    If G is a tree, this will return a dict of positions for a hierarchical layout.

    G: the graph (must be a tree)
    root: the root node
    width: horizontal space allocated for this branch
    vert_gap: gap between levels
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    '''
    def _hierarchy_pos(G, root, left, right, vert_loc, pos=None, parent=None):
        if pos is None:
            pos = {}
        pos[root] = ((left + right) / 2, vert_loc)
        neighbors = list(G.neighbors(root))
        if parent:
            neighbors = [n for n in neighbors if n != parent]
        if neighbors:
            dx = (right - left) / len(neighbors)
            nextx = left
            for neighbor in neighbors:
                next_right = nextx + dx
                pos = _hierarchy_pos(G, neighbor, nextx, next_right, vert_loc - vert_gap, pos, root)
                nextx += dx
        return pos

    return _hierarchy_pos(G, root, 0, width, vert_loc)




def read_payload(file):
    """reads a JSON payload from a file and returns it as a json

    Args:
        file (_type_): file like object, that can be read with open()

    Returns:
        json: a json object of the payload
    """
    with open(file) as text:
        content = text.read()
        payload = json.loads(content)
    return payload


    
##* UTILS FOR VECTOR CALCULATIONS 

def get_cosym(vec_1, vec_2):
    en = np.inner(vec_1,vec_2)
    den = np.linalg.norm(vec_1)*np.linalg.norm(vec_2)
    return en/den


##? is this necessary??
def create_embedding(text:str):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding
    
    
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

def get_all_props_from_payload(payload: dict):
    """looks for all keys in a nested json

    Args:
        payload (json or dict): _description_

    Returns:
        _type_: _description_
    """
    props = []
    for k, v in payload.items():
        if isinstance(v, dict):
            props.extend(get_all_props_from_payload(v))
        else:
            props.append(k)
    return props
    

def get_prop_value_pairs(payload: dict):
    """looks for the properties in a nested json, that carry concrete values

    Args:
        payload (dict): a nested json to be parsed

    Returns:
        dict: a dictionary containing all key-value pairs of the nested json, where the values are not a json
    """
    pairs = []
    for k, v in payload.items():
        if isinstance(v, dict):
            pairs.extend(get_prop_value_pairs(v))
        elif isinstance(v, list):
            for i in v:
                pairs.extend(get_prop_value_pairs(i))
        else:
            pair = {k:v}
            pairs.append(pair)
    return pairs
    
    
def create_nested_list(list_of_dict):
    result_list=[]

    def iterate_nested_structure(nested_struct, count=0, parent_key=['root'], level=0):
        """
        Recursively iterates through a nested data structure (which can include dictionaries and lists), logging each leaf node along with its path. The function is designed to handle complex nested structures, ensuring that each path is accurately recorded in the log file.

        Parameters:
            nested_struct (dict or list): The nested structure to be iterated. This can be a dictionary, list, or a combination of both.
            count (int, optional): A counter for the leaf nodes encountered during the iteration. Defaults to 0.
            parent_key (list, optional): A list tracking the path of keys or indices leading to the current item. This defaults to ['root'] as the starting point.
            level (int, optional): Tracks the current depth level in the nested structure. It is used internally to manage the recursion depth and proper path handling.

        Returns:
            tuple: A tuple containing the final count of leaf nodes, the reset `parent_key` list (which is reset to ['root']), and the current level (used internally for recursion).

        The function writes each leaf node's path and value to a log file using the Python `logging` module. The log entries reflect the hierarchical structure of the nested data, providing a clear trace of each element's location within the nested structure. 
        The function handles both non-empty and empty elements, logging the latter as 'empty'.
        """
        if isinstance(nested_struct, dict):
            for key, value in nested_struct.items():
                if isinstance(value, (dict, list)):
                    parent_key.append(key)
                    level+=1
                    count, parent_key, level = iterate_nested_structure(value, count, parent_key,level)
                    if(len(parent_key)>1):  
                        parent_key.pop()
                else:
                    count += 1
                    output = ' -> '.join(list(map(str, parent_key))) + ' -> ' + str(key)
                    
                    concatenation=parent_key+[str(key)]+[str(value)]
                    result_list.append(concatenation)
        else:
            length_list=len(nested_struct)
            if(length_list>0):
                for i in range(len(nested_struct)):
                    if isinstance(nested_struct[i], (dict, list)):
                        parent_key.append(str(i))
                        level+=1
                        count, parent_key, level = iterate_nested_structure(nested_struct[i], count, parent_key,level)
                        parent_key.pop()
                    else:
                        count += 1 
                        output = ' -> '.join(list(map(str, parent_key)))
                        
                        concatenation=parent_key+[str(nested_struct[i])]
                        result_list.append(concatenation)
            else:
                count += 1 
                output = ' -> '.join(list(map(str, parent_key)))
                concatenation=parent_key+['empty']
                result_list.append(concatenation)
        if(level==1):
            parent_key = ['root']
        level-=1
        return count, parent_key, level
    for s in list_of_dict:
        iterate_nested_structure(s)
    return result_list




def in_database(collection, uri):
    """Check if a given URI is already in the database.

    Args:
        uri (str): The URI to check.

    Returns:
        bool: True if the URI is in the database, False otherwise.
    """
    result = collection.get(
        ids = uri
    )
    return result["ids"] != []


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
                case OWL.unionOf:                                   ## NOTE: stored as list
                    union_blank = op_quad[0].object
                    # print(f" in union case, quad: {union_blank}")
                    popped = extract_rdf_list(union_blank, store)
                case OWL.intersectionOf:                            ## NOTE: stored as list 
                    intersection_blank = op_quad[0].object
                    # print(f" in intersection case, quad: {intersection_blank}")
                    popped = extract_rdf_list(intersection_blank, store)
                case OWL.complementOf:                              ## NOTE: points to class
                    popped = [op_quad[0].object]
                    # print(f" in complement case, quad: {popped}")
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
                case _:
                    return list(blank_quads), op_type
        except Exception as e:
            raise TypeError(f"some error: {str(e)}")
    elif len(op_types) == 0:
        # print(len(blank_quads))
        # for q in blank_quads:
        #     print(q)
        raise ValueError(f"no operator type detected")
    else:
        raise ValueError("multiple operator types detected in one blank node")
    return popped, op_type


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


def get_json_graph(payload: dict) -> nx.DiGraph:
    """creates a nx.Digraph instance from a nested json payload

    Args:
        payload (dict): the jsonpayload as a dictionary (json)

    Returns:
        nx.DiGraph: the payload as a nx.DiGraph
        
    Example:
    ```python
    import json
        
    payload = "path/to/file"
    json_data = json.loads(payload)
    graph = get_json_graph(json_data)
    ```
    """
    graph = nx.DiGraph()
    counter = 0
    current = [(counter, payload)]  
    graph.add_node(counter)
    counter += 1

    while current:
        parent_id, step = current.pop()
        for k, v in step.items():
            edge_data = {"label": preprocess_pascalCase(k)}
            if isinstance(v, dict):
                node_id = counter
                counter += 1
                graph.add_node(node_id)
                current.append((node_id, v))
                graph.add_edge(parent_id, node_id, **edge_data)
            elif isinstance(v, list):
                ## TODO: IMPLEMENT PROPER LIST ADDITION TO GRAPH
                pass
            else:
                leaf_id = counter
                counter += 1
                node_data = {"value": v, "type": type(v)}
                graph.add_node(leaf_id, **node_data)
                graph.add_edge(parent_id, leaf_id, **edge_data)
    return graph
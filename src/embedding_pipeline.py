import networkx as nx
from pyoxigraph import *
import json
import chromadb
import time
from pprint import pprint
from node2vec import Node2Vec


def get_classes(store : Store)->dict:
    
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    
    SELECT DISTINCT ?class ?label
    WHERE{
        
        ?class rdf:type owl:Class;
                rdfs:label ?label;
        }
    """   
    results = store.query(query)
    serialized = results.serialize(format=QueryResultsFormat.JSON)
    jay = json.loads(serialized)
    classes = []
    for c in jay["results"]["bindings"]:##! this could be handled through multiparallelism on class batches
        class_dict = {
            "iri": c["class"]["value"],
            "label": c["label"]["value"]
        }
        class_node = NamedNode(c["class"]["value"])
        parents = []
        soc = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
        subClassResults = store.quads_for_pattern(class_node, soc, None, None)
        for r in subClassResults:
            o = r.object
            if type(o) == NamedNode:
                parents.append(o.value) ##! Not quite what i want in otput
        class_dict["parents"] = parents
        classes.append(class_dict)
    return classes

def get_object_properties(store : Store) -> list:
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    
    SELECT ?prop ?label 
    WHERE{
        
        ?prop rdf:type owl:ObjectProperty;
              rdfs:label ?label.
    }
    """
    results = store.query(query)
    serialized = results.serialize(format=QueryResultsFormat.JSON)
    jay = json.loads(serialized)
    properties = []
    for p in jay["results"]["bindings"]:
        prop_dict = {
            "iri:": p["prop"]["value"],
            "label": p["label"]["value"],
        }
        dom = []
        ran = []
        p_node = NamedNode(p["prop"]["value"])
        dom_node = NamedNode("http://www.w3.org/2000/01/rdf-schema#domain")
        ran_node = NamedNode("http://www.w3.org/2000/01/rdf-schema#range")
        dom_results = store.quads_for_pattern(p_node, dom_node, None, None)
        for r in dom_results:
            if type(r.object) == NamedNode:
                    dom.append(r.object.value)
            elif type(r.object) == BlankNode:
                    pass
        prop_dict["dom"] = dom
        del dom_results
        ran_results = store.quads_for_pattern(p_node, ran_node, None, None)
        for r in ran_results:
            if type(r.object) == NamedNode:
                    ran.append(r.object.value)
            elif type(r.object) ==BlankNode:
                    pass
        prop_dict["ran"] = ran
        properties.append(prop_dict)
    return properties


def embed_classes(classes:list, collection: chromadb.Collection) -> None:
    """embeds the lables of all classes using chroma db

    Args:
        bindings (list): bindings from a sparqlquery returning classes
        collection (_type_): the collection to store the embeddings
    """
    start_time = time.time()
    for c in classes:
        iri = c["iri"]
        label = c["label"]
        label_collection.add(
            ids = [iri],
            documents = [label]
            
        )
    duration = time.time() - start_time
    print(f"embedding time for classes: {duration}")

def embed_props(bindings:list, collection : chromadb.Collection) -> None:
    start_time = time.time()
    for b in bindings:
        iri = b["prop"]["value"]
        label = b["label"]["value"]
        label_collection.add(
            ids = [iri],
            documents = [label]
        )
    duration = time.time() - start_time
    print(f"embedding time for properties: {duration}")
    
def get_prop_dom_ran(prop_bindings:dict, class_bindings: dict)->tuple:
    """returns a list of all classes, that can carry a given property

    Args:
        prop_bindings (dict): the bindings given from the SPARQL Query
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        
        SELECT ?prop ?label ?domain ?range
        WHERE{
            
            ?prop rdf:type owl:ObjectProperty;
                rdfs:label ?label.
            
            OPTIONAL
            {
            ?prop rdfs:domain ?domain;
                rdfs:range ?range.
            }
        }
        as specified by the SPARQL Query language

    Returns:
        tuple of lists: dom, ran each containing all classes that be in domain or range respectively of the given property
    """
    dom = []
    ran = []
    match prop_bindings["dom"]["type"]:
        case "uri":
            dom += [prop_bindings["dom"]["value"]]
            
            
    return dom, ran


def create_core_graph(classes:list) -> nx.DiGraph:
    start_time = time.time()
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (c["iri"], {"label": c["label"]}) for c in classes
    )
    for c in classes:
        parents = c["parents"] ##! does not handle multiple inheritance yet
        graph.add_edges_from([
            (c["iri"], parent, {
                "iri": "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                "label": "sub class of"
            })
            for parent in parents
        ])

    print(f"")
    return graph

def create_dom_ran_graph(core_graph:nx.Graph, prop_bindings: list)->nx.Graph:
    for p in props:
        pass
    print(f"dom-ran-graph creation : {duration}")
    return none

def embed_graph(graph: nx.DiGraph)->None:
    g_time = time.time()
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print(f"graph embedding time : {time.time()-g_time}")

def embed_ontology() -> None:
    
    store = Store()
    client = chromadb.Client()
    label_collection = client.create_collection("label")
    structure_collection = client.create_collection("structure") 
    
    start_time = time.time()
    class_bindings = get_classes(store)
    prop_bindings = get_object_properties(store)
    graph = create_graph(class_bindings, prop_bindings)
    embed_classes(class_bindings)
    embed_props(prop_bindings)
    embed_graph(graph)
    
    print(f"total runtime: {time.time()-start_time}")


if __name__ == "__main__":
    store.load(path = "/home/kipp_sa/github/EmbedAlign/test/bfo-core.owl", format = RdfFormat.RDF_XML)
    main()

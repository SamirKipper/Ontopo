import networkx as nx
from pyoxigraph import *
import json
import chromadb
import time
from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from concurrent.futures import ThreadPoolExecutor

from Nodes import *
from Edges import *
from utils import get_dom_ran, format_chroma_results





## NOTE: classes are currently still stored with iri, 

def get_classes(store : Store)->dict:
    start_time = time.time()
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
                parents.append(o.value)
        class_dict["parents"] = parents
        classes.append(class_dict)
    print(f"class retrieval: {time.time() - start_time}")
    return classes

def get_object_properties(store : Store) -> list:
    start_time = time.time()
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
        p_node = NamedNode(p["prop"]["value"])
        prop_dict = {
            "node": p_node,
            "label": p["label"]["value"],
        }
        dom = []
        ran = []
        dom_node = NamedNode("http://www.w3.org/2000/01/rdf-schema#domain")
        ran_node = NamedNode("http://www.w3.org/2000/01/rdf-schema#range")
        dom_results = store.quads_for_pattern(p_node, dom_node, None, None)
        for r in dom_results:
            dom.append(r.object)
            # if type(r.object) == NamedNode:
            #         dom.append(NamedNode(r.object.value))
            # elif type(r.object) == BlankNode:
            #         dom.append(BlankNode(r.object.value))

        prop_dict["dom"] = dom
        del dom_results
        ran_results = store.quads_for_pattern(p_node, ran_node, None, None)
        for r in ran_results:
            ran.append(r.object)
            # if type(r.object) == NamedNode:
            #         ran.append(r.object.value)
            # elif type(r.object) ==BlankNode:
            #         ran.append(BlankNode(r.object.value))
        prop_dict["ran"] = ran
        properties.append(prop_dict)
    print(f"prop retrieval: {time.time() - start_time}")
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
        iri = b["node"].value
        label = b["label"]
        label_collection.add(
            ids = [iri],
            documents = [label]
        )
    duration = time.time() - start_time
    print(f"embedding time for properties: {duration}")
    
    

def set_prop_dom_ran(props: dict, store: Store)->tuple:
    start_time = time.time()
    for p in props:
        dom = p["dom"]
        ran = p["ran"]
        for d in dom:
            if type(d) == NamedNode:
            
                popped_dom = get_dom_ran(poppable = d, store = store) #! overwrite
                p["dom"] = popped_dom
            else:
                pass
        for r in ran:
            if type(r) == NamedNode:
                
                popped_ran = get_dom_ran(poppable = r, store = store) #! overwrite
                p["ran"] = popped_ran
            else:
                pass
    print(f"pop time = {time.time () - start_time }")
            
    return props


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

    print(f"core graph creation : {time.time()-start_time}")
    return graph

def create_dom_ran_graph(core_graph:nx.Graph, prop_bindings: list)->nx.Graph:
    start = time.time()
    for p in prop_bindings:
        try:
            zipped = list(zip(p["dom"], p["ran"]))
            core_graph.add_edges_from(zipped, label = p["node"].value)
        except:
            pass
        
    print(f"dom-ran-graph creation : {time.time()-start}")
    return core_graph




def embed_graph(graph: nx.DiGraph)->None:
## NOTE:
# - 8 workers ~ 29 sec
# - 6 workers ~ 27 sec
# - 4 workers ~ 23 sec
# - 2 workers ~ 20 sec
# - 1 worker  ~ 17 sec
    g_time = time.time()
    node2vec = Node2Vec(graph, dimensions=128, walk_length=5, num_walks=200, workers=1)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embeddings = {key: model.wv[key] for key in model.wv.index_to_key}
    print(f"graph embedding time : {time.time()-g_time}")
    return embeddings

def add_graph_structure(embeddings: dict, collection : chromadb.Collection) -> None:
    ids = list(embeddings.keys())
    vectors = [embeddings[node_id] for node_id in ids]
    documents = [node_id for node_id in ids]
    metadatas = [{"type": "graph_node", "node_id": node_id} for node_id in ids]
    collection.add(
    ids=ids,
    embeddings=vectors,
    documents=documents,
    metadatas=metadatas
    )
    

def embed_ontology(oxi_store, label_collection, structure_collection) -> None:
    class_bindings = get_classes(oxi_store)
    prop_bindings = get_object_properties(oxi_store)
    
    graph = create_core_graph(class_bindings)
    full_graph = create_dom_ran_graph(graph, prop_bindings)
    
    with ThreadPoolExecutor(max_workers = 12) as exec:
        
        f1 = exec.submit(embed_classes ,class_bindings, label_collection)
        f2 = exec.submit(embed_props, prop_bindings, label_collection)
        f3 = exec.submit(embed_graph, full_graph)
    result = f3.result()
    add_graph_structure(result, structure_collection)


if __name__ == "__main__":
    start_time = time.time()
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    store = Store()
    store.load(path = "/home/kipp_sa/github/EmbedAlign/test/bfo-core.owl", format = RdfFormat.RDF_XML)
    Client = chromadb.Client()
    label_collection = Client.get_or_create_collection(name="labels")
    structure_collection = Client.get_or_create_collection(name = "Structure")
    
    embed_ontology(store, label_collection, structure_collection)
    print(f"total runtime: {time.time()-start_time}")
    results = label_collection.query(query_texts = ["time"], n_results = 3)
    formatted = format_chroma_results(results)
    for f in formatted:
        print(f)
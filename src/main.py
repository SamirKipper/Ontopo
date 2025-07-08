import rdflib
import owlapy
import networkx as nx
from pyoxigraph import *
import json
import chromadb
import time
from pprint import pprint
from node2vec import Node2Vec

## TODO:
    # - [ ] Find way of querying a blank node for disjoints and blank node ranges and domains, currently filtered

store = Store()
client = chromadb.Client()
label_collection = client.create_collection("label")
structure_collection = client.create_collection("structure") 

def get_classes()->dict:
    
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    
    SELECT DISTINCT ?class ?label ?parent
    WHERE{
        {
        ?class rdf:type owl:Class;
                rdfs:label ?label;
                rdfs:subClassOf ?parent.
        }
    UNION {
        ?class rdf:type rdfs:Class;
                rdfs:label ?label;
                rdfs:subClassOf ?parent.
            }
    FILTER(!isBlank(?parent))
    }
    
    """   
    results = store.query(query)
    serialized = results.serialize(format=QueryResultsFormat.JSON)
    jay = json.loads(serialized)
    return jay["results"]["bindings"]

def get_object_properties()->dict:
    query = """
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
    """
    results = store.query(query)
    serialized = results.serialize(format=QueryResultsFormat.JSON)
    jay = json.loads(serialized)
    return jay["results"]["bindings"]


def embed_classes(bindings:list, collection= label_collection) -> None:
    start_time = time.time()
    for b in bindings:
        iri = b["class"]["value"]
        label = b["label"]["value"]
        label_collection.add(
            ids = [iri],
            documents = [label]
            
        )
    duration = time.time() - start_time
    print(f"embedding time for classes: {duration}")

def embed_props(bindings:list, collection = label_collection) -> None:
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

## NOTE: handling embedding in separate function will make the embedding process extremely slow due to multiple iterations
## TODO: concat!
def create_graph(classes:list, props:list) -> nx.DiGraph:
    start_time = time.time()
    graph = nx.DiGraph()
    graph.add_nodes_from(
        (c["class"]["value"], {"label": c["label"]["value"]}) for c in classes
    )
    for c in classes:
        class_iri = c["class"]["value"]
        parent = c["parent"]["value"]
        graph.add_edge(class_iri, parent, **{"iri":"http://www.w3.org/2000/01/rdf-schema#subClassOf", "label":"sub class of"})
    for p in props:
        if "domain" in p.keys() and p["domain"]["type"] == "uri":
            dom = p["domain"]["value"]
        else:
            dom = None                   
        if "range" in p.keys() and  p["range"]["type"] == "uri":
            ran = p["range"]["value"]
        else:
            dom = None
        if dom and ran:
            graph.add_edge(dom,ran, **{"iri":p["prop"]["value"], "label": p["label"]["value"] })
        duration = time.time()-start_time
    print(f"graph creation : {duration}")
    return graph


def main() -> None:
    start_time = time.time()
    class_bindings = get_classes()
    prop_bindings = get_object_properties()
    embed_classes(class_bindings)
    embed_props(prop_bindings)
    graph = create_graph(class_bindings, prop_bindings)
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    print(f"runtime: {time.time()-start_time}")


if __name__ == "__main__":
    store.load(path = "/home/kipp_sa/github/EmbedAlign/test/bfo-core.owl", format = RdfFormat.RDF_XML)
    main()


## * Local
from utils import *
## * Dependencies
from nltk.corpus import wordnet
from owlready2 import *
import chromadb
import sknetwork as skn
import networkx as nx

## TODO:
    # - restucture dicts so you can use them in a plot


def get_graph(ontology: Ontology | World): 
    g = nx.Graph(directed = True)
    g.add_nodes_from((c, {"uri":c.iri, "label": c.label[0]}) for c in ontology.classes())
    for c in ontology.classes():
        for sub in c.subclasses():
            if type(sub) == owlready2.entity.ThingClass:
                g.add_edge( sub, c, label = "http://www.w3.org/2000/01/rdf-schema#subClassOf")
        for l in c.disjoints():
            for d in l.entities:
                if d.iri == c.iri:
                    pass
                else:
                    if not g.has_edge(c,d):
                        g.add_edge(d,c, label = "http://www.w3.org/2002/07/owl#disjointWith")
                    else:
                        pass
                        
    return g









if __name__ == "__main__":
    
    
    
    db = chromadb.Client()
    label_collection = db.create_collection(name = "labels")
    definitions = db.create_collection(name = "definitions")
    structure = db.create_collection(name = "structure")







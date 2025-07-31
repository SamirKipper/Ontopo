import os
import time

from pyoxigraph import *
import chromadb
import networkx as nx

from OWL import OWL
from RDFS import RDFS
from Nodes import *
from Edges import *
from utils import *

def get_classes(store: Store):
    class_gen = get_all_classes(store)
    classes = []
    for c in class_gen:
        if isinstance(c, NamedNode):
            classes += [NamedClass(c.value, store)]
        elif isinstance(c, BlankNode):
            popped , op_type = pop_blank(c, store)
            match op_type:
                case OWL.unionOf:
                    classes += [UnionClass(c.value, popped, store)]
                case OWL.intersectionOf:
                    classes += [IntersectionClass(c.value, popped, store)]
                case OWL.complementOf:
                    if len(popped) == 1:
                        classes += [NegationClass(c.value,popped[0], store)]
                    else:
                        raise NotImplementedError
                case _:
                    raise NotImplementedError
    return classes

def create_hierarchy_graph(named: list):
    graph = nx.DiGraph()
    # for n in named:
    #     graph.add_node(n, label = n.RDFSlabel )
    graph.add_nodes_from(named)
    for n in named:
        subclasses = list(n.subClasses)
        if subclasses != []:
            for s in subclasses:
                attrs = {
                        "type": RDFS.subClassOf,
                        "label": "sub class of"
                        }
                graph.add_edge(s, n, **attrs)
    return graph

def create_core_graph(hierarchy: nx.DiGraph, blanks : list) -> nx.DiGraph:
    graph = hierarchy
    graph.add_nodes_from(blanks)
    for b in blanks:
        if isinstance(b, UnionClass):
            edge_data = {
                "type" : OWL.unionOf,
                "label" : "in union"
            }
            for c in b.classes:
                graph.add_edge(c, b, **edge_data)
        if isinstance(b, NegationClass):
            edge_data = {
                "type" : OWL.complementOf,
                "label" : "not in"
            }
            graph.add_edge(b.onClass, b, **edge_data)
        if isinstance(b, IntersectionClass):
            edge_data = {
                "type" : OWL.intersectionOf,
                "label" : "intersection of"
            }
            for c in b.classes:
                graph.add_edge(c, b, **edge_data)
    return graph




if __name__ == "__main__":
    start_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    store = Store()
    store.load(path = "/home/kipp_sa/github/EmbedAlign/test/bfo-core.owl", format = RdfFormat.RDF_XML)
    Client = chromadb.Client()
    label_collection = Client.get_or_create_collection(name="labels")
    structure_collection = Client.get_or_create_collection(name = "structure")
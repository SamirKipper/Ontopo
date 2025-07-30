import os
import time

from pyoxigraph import *
import chromadb
import networkx as nx

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

def create_core_graph(classes: list):
    graph = nx.DiGraph()
    for c in classes:
        graph.add_node(c)
    for c in classes:
        if isinstance(c, NamedClass):
            for s in c.subclasses:
                pass
    return graph




if __name__ == "__main__":
    start_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    store = Store()
    store.load(path = "/home/kipp_sa/github/EmbedAlign/test/bfo-core.owl", format = RdfFormat.RDF_XML)
    Client = chromadb.Client()
    label_collection = Client.get_or_create_collection(name="labels")
    structure_collection = Client.get_or_create_collection(name = "Structure")
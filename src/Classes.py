from pyoxigraph import NamedNode, BlankNode, Store
from utils import *
from typing import Union, List
from OWL import OWL



class IntersectionClass:
    def __init__(
        self,
        blank : BlankNode):
        self.blank = blank
    
    @property
    def classes(self):
        classes = handle_intersection(self.blank)
        return classes
        
    def __repr__(self):
        string = ' & '.join(str(node) for node in self.classes)
        return string

class UnionClass:
    def __init__(
        self,
        classes : List [ Union [ BlankNode , NamedNode ] ]
    ):
        self.classes = classes
        
    def __repr__(self):
        string = ' | '.join(str(node) for node in self.classes)
        return string

class NegationClass:
    def __init__(
                self, 
                node : NamedNode
                ):
        self.node = node
    
    @property
    def entails(self):
        classes = get_complement(self.node)
        return classes


class ComplexClass:
    def __init__(
        self,
        blank : BlankNode,
        store: Store):
        self.blank = blank
        self.store = store
    
    @property
    def tree(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        blanknode = [self.blank]
        graph.add_node(self.blank)
        while blanknode:
            current = blanknode.pop()
            graph.remove_node(current)
            popped, op_type = pop_blank(current, self.store)
            graph.add_node(op_type)
            for p in popped:
                if type(p) == NamedNode:
                    graph.add_node(p)
                    graph.add_edge(p, current)
                if type(p) == BlankNode:
                    blanknode.append(p)
                    graph.add_node(p)
                    graph.add_edge(p,current)
        return graph
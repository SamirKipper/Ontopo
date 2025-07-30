from pyoxigraph import NamedNode, BlankNode, Store
from utils import *
from typing import Union, List
from OWL import OWL


##################################################*
#*                Classtypes                     #  
##################################################*

class NamedClass:
    def __init__(
        self,
        iri : str,
        store : Store,
        
    ):
        self.iri = iri
        self._storeNode = NamedNode(self.iri)
        self.store = store
        
    
    @property
    def RDFSlabel(self):
        quads = self.store.quads_for_pattern(self._storeNode, RDFS.label, None, None)
        labels = [q.object.value for q in quads]
        return labels
    
    @property
    def subClasses(self):
        return ( NamedClass(c.value, self.store) for c in get_subclasses(self._storeNode, self.store) )
    
    def __hash__(self):
        return hash(self.iri)
    
    def __repr__(self):
        return str(self.iri)


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


class IntersectionClass(ComplexClass):
    def __init__(
        self,
        blank : BlankNode,
        classes: list,
        store : Store):
        self.blank = blank
        self.classes = classes
        self.store = store
    
    @property
    def tree(self):
        graph = nx.DiGraph()
        popped, op_type = pop_blank(self.blank, self.store)
        graph.add_node(op_type)
        for p in popped:
            graph.add_node(p)
            graph.add_edge(op_type,p, value = OWL.intersectionOf)
        return graph
    
    def __hash__(self):
        return hash(self.blank)
    
    def __repr__(self):
        string = ' & '.join(str(node) for node in self.classes)
        return string


class UnionClass(ComplexClass):
    def __init__(
        self,
        blank : BlankNode,
        classes: list,
        store : Store,
        
    ):
        self.blank = blank
        self.classes = classes
        self.store = store
        
        
    @property
    def tree(self):
        graph = nx.DiGraph()
        popped, op_type = pop_blank(self.blank, self.store)
        graph.add_node(op_type)
        for p in popped:
            graph.add_node(p)
            graph.add_edge(op_type,p, value = OWL.unionOf)
        return graph
        
    def __hash__(self):
        return hash(self.blank)    
    
    def __repr__(self):
        string = ' | '.join(str(node) for node in self.classes)
        return string


class NegationClass(ComplexClass):
    def __init__(
                self, 
                blank : NamedNode,
                onClass: NamedNode | BlankNode,
                store = Store
                ):
        self.blank = blank
        self.onClass = onClass
        self.store = store
    
    @property
    def entails(self):
        classes = get_complement(self.node)
        return classes
    
    def __hash__(self):
        return hash(self.blank)
    
    def __repr__(self):
        return f"(¬{self.onClass.value})"


class RestrictionClass:
    def __init__(
        self,
        blank : BlankNode,
        onProperty,
    ):
        self.onProperty = onProperty
        self.blank = blank
        
    def __hash__(self):
        return hash(self.blank)


class SOME(RestrictionClass):
    def __init__(
        self, 
        blank, 
        onProperty,
        ):
        super().__init__(blank, onProperty)
    
    def __repr__(self):
        string = str(self.onProperty) + " SOME "
        



#######################################################*
#*                   others                           #*
#*#####################################################*


class Datatype:
    def __init__(
        self,
        iri : str):
        self.iri = iri
        
        
class Individual:
    def __init__(
        self,
        iri : str,
        instance_of : NamedClass | ComplexClass | UnionClass | NegationClass | IntersectionClass):
        self.iri = iri
        self.instance_of = instance_of
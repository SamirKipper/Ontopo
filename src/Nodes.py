from pyoxigraph import NamedNode, BlankNode, Store
from utils import *
from typing import Union, List
from OWL import OWL


##################################################*
#*                Classtypes                     #  
##################################################*


class NamedClass:
    """The Class for all OWL or RDF Classes with an Explicit IRI
    """
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
        """The attribute linking the class to its rdfs:label

        Returns:
            list: the list of rdfs:labels given to the class
        """
        quads = self.store.quads_for_pattern(self._storeNode, RDFS.label, None, None)
        labels = [q.object.value for q in quads]
        return labels
    
    @property
    def subClasses(self):
        quads = self.store.quads_for_pattern(None, RDFS.subClassOf, self._storeNode, None)
        return ( NamedClass(c.subject.value, self.store) for c in quads)
    
    @property
    def descendents(self):
        desc = get_descendents(self._storeNode, self.store)
        return (NamedClass(s, self.store) for s in desc)
    
    def __hash__(self):
        return hash(self.iri)
    
    def __repr__(self):
        return "'" + str(self.RDFSlabel[0]) + "'"
    def __eq__(self, other):
        return isinstance(other,NamedClass) and self.iri == other.iri


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
    
    def __eq__(self, other):
        return self.blank == other.blank

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
        classes = get_complement(self.onClass)
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
        return string


class ONLY(RestrictionClass):
    def __init__(
        self, 
        blank, 
        onProperty,
        ):
        super().__init__(blank, onProperty)
    
    def __repr__(self):
        string = str(self.onProperty) + " ONLY "
        return string


class VALUE(RestrictionClass):
    def __init__(
        self, 
        blank, 
        onProperty,
        ):
        super().__init__(blank, onProperty)
    
    def __repr__(self):
        string = str(self.onProperty) + " VALUE "
        return string


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
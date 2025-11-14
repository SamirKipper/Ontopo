from pyoxigraph import NamedNode, BlankNode, Store
from abc import ABC

from RDF import RDF
from RDFS import RDFS
from OWL import OWL
from Nodes import *



english_tags = ["en", "en-us", "en-gb", "en-au", "en-ca", "en-nz"]

class OntoEdge(ABC):
    pass


class ObjectProperty(OntoEdge):
    def __init__(
        self,
        iri : str,
        store : Store
    ):
        self.iri = iri
        self.store = store
        self.node = NamedNode(self.iri)
        
    @property
    def domain(self) -> NamedNode | BlankNode:
        quads = self.store.quads_for_pattern(self.node, RDFS.domain, None, None)
        dom = (map_class_type(q.object, store = self.store) for q in quads)
        return dom
    
    @property
    def range(self):
        quads = self.store.quads_for_pattern(self.node, RDFS.range, None, None)
        dom = (map_class_type(q.object, store = self.store) for q in quads)
        return dom
    
    @property
    def rdfsLabel(self):
        """The attribute linking the class to its rdfs:label

        Returns:
            list: the list of rdfs:labels given to the class
        """
        quads = self.store.quads_for_pattern(self.node, RDFS.label, None, None)
        labels = [q.object for q in quads]
        return labels
    
    def __repr__(self):
        if (self.rdfsLabel != None) and (len(self.rdfsLabel) > 0):
            english_labels = [l.value for l in self.rdfsLabel if l.language in english_tags]
            if len(english_labels) != 0:
                return english_labels[0]
            elif len(english_labels) == 0:
                return f"{self.rdfsLabel[0].value}"
                
        else:
            return self.iri


class DatatypeProperty(OntoEdge):
    def __init__(
        self,
        iri : str,
        store : Store
    ):
        self.iri = iri
        self.store = store
        
    @property
    def rdfsLabel(self):
        """The attribute linking the class to its rdfs:label

        Returns:
            list: the list of rdfs:labels given to the class
        """
        quads = self.store.quads_for_pattern(self._storeNode, RDFS.label, None, None)
        labels = [q.object for q in quads]
        return labels
    
    def __repr__(self):
        if (self.rdfsLabel != None) and (len(self.rdfsLabel) > 0):
            english_labels = [l.value for l in self.rdfsLabel if l.language in english_tags]
            if len(english_labels) != 0:
                return english_labels[0]
            elif len(english_labels) == 0:
                return f"{self.rdfsLabel[0].value}"
        else:
            return self.iri

class AnnotationProperty(OntoEdge):
    def __init__(
        self,
        iri : str,
    ):
        self.iri = iri
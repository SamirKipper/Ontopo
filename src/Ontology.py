import networkx as nx
from pyoxigraph import Store, RdfFormat
from collections.abc import Iterator
import os
from Ressources import *



def return_formatter(file: str) -> RdfFormat:
    file_extension = os.path.splitext(file)[1].lower()
    match file_extension:
        case ".ttl":  
            formatter = RdfFormat.TURTLE
        case ".rdf":  
            formatter = RdfFormat.RDF_XML
        case ".n3":  
            formatter = RdfFormat.N3
        case ".nt":  
            formatter = RdfFormat.N_TRIPLES
        case ".jsonld":  
            formatter = RdfFormat.JSON_LD
        case ".trig":  
            formatter = RdfFormat.TRIG
        case ".nq":  
            formatter = RdfFormat.N_QUADS
        case ".xml":  
            formatter = RdfFormat.RDF_XML
        case ".owl":  
            formatter = RdfFormat.RDF_XML
        case _:
            raise ValueError(f"Unsupported file format: {file_extension}")
    return formatter




class Ontology:
    """
    #### Overview
    The baseclass for an ontology.
    
    #### properties:
    - store: the store, that the d4c is stored in
    - classes: iterator over all classes of the ontology, including complex classes, loads lazily
    - named_classes: iterator over all classes of the ontology with an iri
    - object_properties: iterator over all object properties in the ontology
    - datatype_properties: iterator over all datatype properties in the ontology
    
    #### methods
    - embed_labels: iterates over the ressources, creates an embedding of the labels if given and stores them in the collection
    - embed_structure: uses the node2vec framework to embed the ontology
    - embed_structure: calls embed_labels and embed_structure in a thread
    
    #### example
    ```python
    store = Store('path/to/file')
    ont = Ontology( store = store)
    for c in ont.named_classes:
        print(c.rdfsLabel)
    ```
    
    """
    
    @classmethod
    def load(cls, file : str) -> "Ontology":
        """loads the ontology from the given file path 

        Args:
            file (str): a file path

        Returns:
            Ontology: the Ontology
        """
        store = Store()
        parser = return_formatter(file)
        store.load(path = file, format=parser)
        return Ontology(file, store)
        
    
    def __init__(self , file : str, store : Store):
        self.file = file
        self.store = store
    
    @property
    def classes(self) -> Iterator[NamedClass | UnionClass | NegationClass | IntersectionClass]:
        """
        ### Overview
        the property accesses the internal database and returns all types of 
        classes from the d4c ontology, including complex classes
        It loads the classes lazily.
        
        ### Returns:
        Iterator[NamedClass | UnionClass | NegationClass | IntersectionClass]: _description_
        """
        quads = self.store.quads_for_pattern(None, RDF.type, OWL.Class, None)
        classes = (q.subject for q in quads)
        for c in classes:
            inst = map_class_type(c, self.store)
            if inst:
                yield inst
            else: 
                quads = self.store.quads_for_pattern(c, None, None, None)
                print("-----------------------------")
                for q in quads:
                    print(q)
    
    @property
    def named_classes(self):
        """returns all classes, that are contained in the Ontology, that have an IRI

        Returns:
            Generator: A Generator over all NamedClasses
        """
        classes = self.store.quads_for_pattern(None, RDF.type, OWL.Class, None)
        return (NamedClass(iri = c.subject.value, store = self.store) for c in classes if isinstance(c.subject, NamedNode))
    
    @property
    def object_properties(self):
        props = self.store.quads_for_pattern(None, RDF.type, OWL.ObjectProperty, None)
        return (ObjectProperty(p.subject.value, store= self.store) for p in props if isinstance(p.subject, NamedNode))
    
    @property
    def datatype_properties(self):
        props = self.store.quads_for_pattern(None, RDF.type, OWL.DatatypeProperty, None)
        return (DatatypeProperty(p.subject.value, store = self.store) for p in props if isinstance(p.subject, NamedNode))
    
    @property ## NOTE: HANDLE LABEL ERRORS HERE, SO ITERATION OVER GRAPH IS SIMPLE
    def structure(self) -> nx.DiGraph:
        ##* CORE GRAPH CREATION
        graph = nx.DiGraph()
        graph.add_nodes_from(self.classes)
        for c in self.classes:
            if isinstance(c, NamedClass):
                data = {"type" : "sub Class"}
                subclasses = c.subClasses
                zipped = zip([c], subclasses)
                graph.add_edges_from(zipped, **data)
            elif isinstance(c, UnionClass):
                data = {"type" : "union of"}
                zipped = zip([c], c.onClasses)
                graph.add_edges_from(zipped, **data)
            elif isinstance(c, IntersectionClass):
                data = {"type" : "intersection of"}
                zipped = zip([c], c.onClasses)
                graph.add_edges_from(zipped, **data)
            elif isinstance(c, NegationClass):
                data = {"type" : "complement of"}
                graph.add_edge(c, c.onClass, **data)
        ##* ADDING EDGES FOR OBJECT PROPERTIES
        for p in self.object_properties:
            dom = list(p.domain)
            ran = list(p.range)
            if dom == []: 
                dom = [OWL.Thing]
            if ran == []:
                ran = [OWL.Thing]
            zipped = zip(dom, ran)
            data = {
                "type" : "object property",
                "iri" : p.iri,
            }
            graph.add_edges_from(zipped, **data)
        return graph
    
    def topoligize(self):
        pass
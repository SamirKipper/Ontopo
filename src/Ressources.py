from pyoxigraph import QuerySolutions, QueryResultsFormat, NamedNode, BlankNode, Store, Literal, Variable
import chromadb
from abc import ABC

from OWL import OWL
from utils import *
from decomposition import *


english_tags = ["en", "en-us", "en-gb", "en-au", "en-ca", "en-nz"]

def map_property_type(node : NamedNode | BlankNode, store : Store):
    if isinstance(node, NamedNode):
        quads = list(store.quads_for_pattern(node, RDF.type, None, None))
        objs  = [q.object for q in quads]
        if OWL.ObjectProperty in objs:
            return ObjectProperty(
                        node.value, store = store
                    )
        elif OWL.AnnotationProperty:
            return AnnotationProperty(
                node.value, store = store
                )
        elif OWL.DatatypeProperty:
            return DatatypeProperty(
                node.value, store = store
            )
        else:
            raise ValueError(f"property noen of object, annotation or datatype property")


def __map_restriction_type(blank, dictio : dict, store : Store):
    if OWL.minCardinality in dictio.keys():
        print("------------------------------------")
    elif OWL.minQualifiedCardinality in dictio.keys():
        return MinQualifiedCardinality(
            store = store,
            onProperty = dictio[OWL.onProperty], 
            onClass = dictio[OWL.onClass], 
            value = dictio[OWL.minQualifiedCardinality] 
            )
    elif OWL.maxCardinality in dictio.keys():
        return MaxCardinality(
            blank = blank,
            store = store,
            value = dictio[OWL.maxCardinality] ,
            onProperty = dictio[OWL.onProperty],
            )
    elif OWL.maxQualifiedCardinality in dictio.keys():
        return MaxQualifiedCardinality(
            store = store,
            onProperty = dictio[OWL.onProperty], 
            onClass = dictio[OWL.onClass], 
            value = dictio[OWL.minQualifiedCardinality] 
            )
    elif OWL.allValuesFrom in dictio.keys():
        if OWL.onProperty in dictio.keys():
            onProperty = dictio[OWL.onProperty]
            return Only(blank = blank, store = store, onProperty = onProperty, onClass = dictio[OWL.allValuesFrom])
        else: 
            raise ValueError("on property missing in a Object Property restriction")
    elif OWL.someValuesFrom in dictio.keys():
        if OWL.onProperty in dictio.keys():
            onProperty = dictio[OWL.onProperty]
            return Some(blank  = blank, store = store, onProperty = onProperty, onClass = dictio[OWL.someValuesFrom])
        else: 
            raise ValueError("on property missing in a Object Property restriction")
    elif OWL.hasValue in dictio.keys():
        if OWL.onProperty in dictio.keys():
            onProperty = dictio[OWL.onProperty]
            return Value(blank  = blank, store = store, onProperty = onProperty)
        else: 
            raise ValueError("on property missing in a Object Property restriction")
    elif OWL.cardinality in dictio.keys():
        onProperty = dictio[OWL.onProperty]
        return Cardinality(blank = blank, store = store, onProperty = onProperty, value = dictio[OWL.cardinality])
    else:
        raise NotImplementedError(f"unkown restriction type in dictionary: {dictio}")


def map_class_type(node : NamedNode | BlankNode, store : Store):
    """
    identifies the class type from the given node and instantiates the respective class.
    Args:
        node (NamedNode | BlankNode): One of the Node types of pyoxigraph
        store (Store): the store to look into
    Raises:
        NotImplementedError: If it finds a class type, that is not implemented
        ValueError: else
    Returns:
        NamedClass | UnionClass, NegationClass of IntersectionClass: the correct type depending on your class
    """
    try:
        unimplemented = []
        if isinstance(node, NamedNode):
            return NamedClass(node.value, store = store)
        elif isinstance(node, BlankNode):
            _ , op_type  = pop_blank(node, store)
            match op_type:
                ###* union, intersection and complement
                case OWL.unionOf:
                    return UnionClass(blank = node, store = store)
                case OWL.complementOf:
                    return NegationClass(blank = node, store = store)
                case OWL.intersectionOf:
                    return IntersectionClass(blank = node, store = store)
                ## * restrictions
                case OWL.onProperty:
                    dictio = pop_blank_to_dict(blanknode = node, store = store)
                    return __map_restriction_type(blank = node, dictio = dictio, store = store)
                case OWL.Restriction:
                    dictio = pop_blank_to_dict(blanknode = node, store = store)
                    return __map_restriction_type(blank = node, dictio = dictio, store = store)
                case OWL.cardinality:
                    dictio = pop_blank_to_dict(blanknode = node, store = store)
                    return __map_restriction_type(blank = node, dictio = dictio, store = store)
                ##* lists
                case OWL.oneOf:
                    class_list = extract_rdf_list(list_node = node, store = store)
                    return OneOf(node, store)
                case RDF.type:
                    pass
                case _:
                    # raise NotImplementedError(f"Class '{op_type}' has not been implemented yet.")
                    # print("-----------------")
                    # for p in popped:
                    #     print(p)
                    if op_type in unimplemented:
                        pass
                    else:
                        unimplemented += [op_type]
            print(unimplemented)
        else: 
            raise ValueError(f"unknown type found at {node} of type  {type(node)}")
    except Exception as e:
        raise ValueError(str(e))


##################################################*
#*                Classtypes                     #*  
##################################################*


class NamedClass:
    """
    ## NamedClass (class)
    The Class for all OWL or RDF Classes with an Explicit IRI  
    
    ### required attributes  
    - iri : string  
    - store : pyoxigraph.Store  
    
    ### computed attributes
    - node : returns a pyoxigraph.NamedNode instance for the value in self.iri
    - rdfsLabes : returns a list of all strings connected to class via rdfs:label
    - subclasses : returns a list of all subclasses of the class  
    
    ### methods
    - embed_label : runs a vector embedding model on the label of the class
    
    ### magic methods implemented
    - __hash__ : returns hash of iri
    - __str__ : returns the first value of rdfslabel if this is not empty, else iri
    - __eq__ : returns true if other iri is identical, else false
    """
    def __init__(
        self,
        iri : str,
        store : Store
        
    ):
        self.iri = iri
        self.node = NamedNode(self.iri)
        self.store = store
        
    @property
    def rdfsLabel(self):
        """The attribute linking the class to its rdfs:label
        Returns:
            list: the list of rdfs:labels given to the class
        """
        try:
            quads = self.store.quads_for_pattern(self.node, RDFS.label, None, None)
            labels = [q.object for q in quads]
        except Exception as e:
            raise ValueError(f"error: {str(e)}")
        return labels
    
    @property
    def subClasses(self):
        """ loads the subclasses of a class lazily

        Returns:
            Generator: a generator over the instantiated classes
        """
        quads = self.store.quads_for_pattern(None, RDFS.subClassOf, self.node, None)
        return ( map_class_type(c.subject, self.store) for c in quads)
    
    @property
    def subClassOf(self):
        """ loads the subclasses of a class lazily

        Returns:
            Generator: a generator over the instantiated classes
        """
        quads = self.store.quads_for_pattern( self.node, RDFS.subClassOf, None, None)
        return (map_class_type(c.object, self.store) for c in quads)
    
    @property
    def descendents(self):
        """ loads the descendents (subclasses of subclasses etc) of a class lazily
        Returns:
            Generator: a generator over the instantiated classes of descendents
        """
        desc = get_descendents(self.node, self.store)
        return (NamedClass(s, self.store) for s in desc)
    
    @property
    def annotations(self):
        
        query = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT  ?p ?ann
        WHERE{{
            <{self.iri}> ?p ?ann.
            ?p rdf:type owl:AnnotationProperty.    
        }}
        """
        results : QuerySolutions = self.store.query(query)
        anns = {}
        for r in results:
            p, ann = r
            anns[p] = ann.value
        return anns
        
    
    def isSubclassOf(self, other) -> bool:
        """method, to check if the class is a subclass of another

        Args:
            other (NamedClass, ComplexClass, RestrictionClass): a class to check, if the class is the subclass of it

        Returns:
            bool: wether or not it is a subclass of other
        """
        results = self.store.quads_for_pattern(self.node, RDFS.subClassOf, None, None)
        r_list = [map_class_type(c) for c in results]
        if other in r_list:
            return True
        else:
            return False
    
    def get_initial_region(self):
        label = str(self)
        sentences = []
        
        subs = list(self.subClasses)
        if subs != []:
            sentence = f"{str(self)} has the subclasses"
            length = len(subs)
            if length >1:
                for i in range(length):
                    if i+1 != length: 
                        sentence += f" {str(subs[i])},"
                    else:
                        sentence += f" and {str(subs[i])}."
                sentences.append(sentence)
            else:
                sentence = f"{str(self)} has the subclasses {subs[0]}"
                sentences.append(sentence)
        else:
            sentences.append(f"{label} has no subclasses")
        pars = list(self.subClassOf)
        if pars != []:
            sentence = f"{str(self)} is a subclass of "
            length = len(pars)
            for i in range(length):
                if i+1 != length: 
                    sentence += f" {str(pars[i])},"
                else:
                    sentence += f" and {str(pars[i])}."
            sentences.append(sentence)
        else:
            sentences.append(f"{label} has no parent classes")
        for a in self.annotations:
            sentences.append(f"{str(self)} has the annotation {str(a)}")
        try:
            center, cov = get_center_and_cov(label, sentences)
        except Exception as e:
            raise e
        return center, cov
    
    def __hash__(self):
        return hash(self.iri)
    
    
    
    def __str__(self):
        try:
            
            if len(self.rdfsLabel) == 0:
                    return str(self.iri)
            else:
                english_labels = [l.value for l in self.rdfsLabel if l.language in english_tags]
                if english_labels != []:
                    return english_labels[0]
                else:
                    return self.rdfsLabel[0].value
        except Exception as l:
            raise ValueError("no label given")
    
    def __eq__(self, other):
        return isinstance(other,NamedClass) and self.iri == other.iri
    
    
        


class ComplexClass:
    def __init__(
        self,
        blank : BlankNode,
        store : Store
        ):
        self.blank = blank
        self.store = store
    
    def __hash__(self):
        return hash(self.blank)


class IntersectionClass(ComplexClass):
    def __init__(
                self,
                blank : BlankNode,
                store : Store
                ):
        self.blank = blank
        self.store = store
    
    @property
    def onClasses(self):
        popped, _ = pop_blank(self.blank, self.store)
        mapped = (map_class_type(p, self.store) for p in popped)
        for m in mapped:
            if m:
                yield m
            else:
                for p in popped:
                    print(p)
                raise ValueError("intersection with none type")
    
    def __eq__(self, other):
        return set(list(self.onClasses)) == set(list(other.onClasses))
    
    def __str__(self):
        string = "(" + ' AND '.join(str(node) for node in self.onClasses) + ")"
        return string


class UnionClass(ComplexClass):
    def __init__(
        self,
        blank : BlankNode,
        store : Store
    ):
        self.blank = blank
        self.store = store

    @property
    def onClasses(self):
        popped, _ = pop_blank(self.blank, self.store)
        mapped = (map_class_type(p, self.store) for p in popped)
        for m in mapped:
            if m:
                yield m
            else:
                raise ValueError(f"union with none type at {m} , mapped : {mapped}")  
    
    def __eq__(self, other):
        set(list(self.onClasses)) == set(list(other.onClasses))
    
    def __str__(self):
        string = "("+' OR '.join(str(node) for node in self.onClasses) + ")"
        return string


class NegationClass(ComplexClass):
    def __init__(
                self, 
                blank : BlankNode,
                store : Store
                ):
        super().__init__(blank, store)
    
    @property
    def onClass(self):
        popped, _ = pop_blank(self.blank, self.store)
        if popped:
            if len(popped) == 1:
                return map_class_type(popped[0], self.store)
            else:
                raise ValueError("found negation class with multiple classes with union or intersection")
        else:
            raise AttributeError("no class set")
    
    @property
    def entails(self):
        classes = get_complement(self.onClass)
        return classes
    
    def __eq__(self, other):
        return set(list(self.onClass)) == set(list(other.onClass))
    
    def __str__(self):
        return f"(NOT {self.onClass})"


class RestrictionClass:
    def __init__(
        self,
        blank : BlankNode,
        store : Store,
        onProperty,
        value : int | None = None
    ):
        self.blank = blank
        self.onProperty = map_property_type(onProperty, store = store)
        self.store = store
        self.value = value
    
    def __hash__(self):
        return hash(self.blank)


class Some(RestrictionClass):
    def __init__(
        self, 
        blank : BlankNode, 
        store : Store,
        onProperty,
        onClass
        ):
        super().__init__(blank = blank,  store = store, onProperty = onProperty)
        self.onClass = map_class_type(onClass, store = self.store)
    
    def __str__(self):
        string = str(self.onProperty) + " SOME " + str(self.onClass)
        return string


class Only(RestrictionClass):
    def __init__(
        self, 
        blank : BlankNode,
        store : Store,
        onProperty,
        onClass
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
        self.store = store
        self.onClass = map_class_type(onClass, store = self.store)
    
    def __str__(self):
        string = "(" + str(self.onProperty) + " ONLY " +  str(self.onClass) + ")"
        return string


class Value(RestrictionClass):
    def __init__(
        self, 
        blank : BlankNode,
        store : Store ,
        onProperty,
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
    
    def __str__(self):
        string = str(self.onProperty) + " VALUE "
        return string

class MinCardinality(RestrictionClass):
    def __init__(
        self,
        blank : BlankNode,
        store : Store,
        onProperty,
        value : int
        
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
        self.value = value
    
    def __str__(self):
        return self.onProperty + "MIN" + str(self.value)

class MinQualifiedCardinality(MinCardinality):
    def __init__(
        self,
        blank : BlankNode,
        store : Store,
        onProperty,
        value : int,
        onClass : NamedNode | BlankNode
        
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty, value = value)
        self.onClass = map_class_type(onClass)
    
    def __str__(self):
        return f"(MIN {self.value} {self.onClass})"

class MaxCardinality(RestrictionClass):
    def __init__(
        self,
        blank : BlankNode,
        store : Store,
        onProperty,
        value : int
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty, value = value)
    
    def __str__(self):
        return f"(MAX {self.value})"

class MaxQualifiedCardinality(MaxCardinality):
    def __init__(
        self,
        blank : BlankNode,
        store : Store,
        onClass : NamedNode | BlankNode, 
        onProperty,
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
        self.onClass = map_class_type(onClass)
    def __str__(self):
        return f"(MAX {self.value})"





class Exactly(RestrictionClass):
    def __init__(
        self,
        blank : BlankNode, 
        store : Store,
        value : int,
        onProperty
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
        self.value = value
    
    def __str__(self):
        return f"({self.onProperty} EXACTLY {self.value})"


class Cardinality(RestrictionClass):
    def __init__(
        self,
        blank : BlankNode, 
        store : Store,
        value : int,
        onProperty
        ):
        super().__init__(blank = blank, store = store, onProperty = onProperty)
        self.value = value
    
    def __str__(self):
        return f"({self.onProperty} CARDINALITY {self.value})"
    
    

class OneOf(ComplexClass, NamedClass):
    """
    ### OneOf (Class)
    the class to instantiate, if an anonymous owl:oneOf class was created.
    Instead of giving a class a name and describing the class via restrictions,
    oneOf can creates an anonymous class, that is the parent for a given list of individuals

    ### necessary attributes
    - node : pyoxigraph.BlankNode or pyoxigraph.NamedNode
    - store : pyoxigraph.Store
    
    ### computed attributes
    - on_instances : return the list of instances
    
    """
    def __init__(
        self,
        node : BlankNode | NamedNode,
        store : Store,
        ):
        self.node = node,
        self.store = store
        
    @property
    def on_instances(self):
        quads = list(self.store.quads_for_pattern(self.node, OWL.oneOf, None, None))
        if len(quads) == 1:
            instance_list = extract_rdf_list(quads[0].object)
            return instance_list
        else:
            raise AttributeError("one of multiple")
        
    def __str__(self):
        return f" one of {self.on_instances}"

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



#######################################################*
#*                   PROPS                            #*
#*#####################################################*

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
    
    def __str__(self):
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
    
    def __str__(self):
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
        store : Store
    ):
        self.iri = iri
        self.store = store
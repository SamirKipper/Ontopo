from pyoxigraph import NamedNode

class RDFS:
    Resource = NamedNode("http://www.w3.org/2000/01/rdf-schema#Resource")
    Class = NamedNode("http://www.w3.org/2000/01/rdf-schema#Class")
    Literal = NamedNode("http://www.w3.org/2000/01/rdf-schema#Literal")
    Datatype = NamedNode("http://www.w3.org/2000/01/rdf-schema#Datatype")
    subClassOf = NamedNode("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    subPropertyOf = NamedNode("http://www.w3.org/2000/01/rdf-schema#subPropertyOf")
    domain = NamedNode("http://www.w3.org/2000/01/rdf-schema#domain")
    range = NamedNode("http://www.w3.org/2000/01/rdf-schema#range")
    label = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
    comment = NamedNode("http://www.w3.org/2000/01/rdf-schema#comment")
    seeAlso = NamedNode("http://www.w3.org/2000/01/rdf-schema#seeAlso")
    isDefinedBy = NamedNode("http://www.w3.org/2000/01/rdf-schema#isDefinedBy")
    containerMembershipProperty = NamedNode("http://www.w3.org/2000/01/rdf-schema#ContainerMembershipProperty")
    Container = NamedNode("http://www.w3.org/2000/01/rdf-schema#Container")
    member = NamedNode("http://www.w3.org/2000/01/rdf-schema#member")
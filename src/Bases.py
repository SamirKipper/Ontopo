from neomodel import StructuredNode, RelationshipTo, StringProperty, StructuredRel, ArrayProperty, FloatProperty

# TODO:
    # - check for proper handling of instantiation and subPropertyOf/subClassOf relationship

class rdfProperty(StructuredRel):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    definition = StringProperty()
    # domain = RelationshipTo('Class', 'DOMAIN')
    # range = RelationshipTo('Class', 'RANGE')
    # subPropertyOf = RelationshipTo('ObjectProperty', 'subPropertyOf')
    

## * PROPERTIES
class ObjectProperty(rdfProperty):
    pass
    
### * Class relationships

class rdfsSubClassOf(rdfProperty):
    pass    

class owlDisjointwith(rdfProperty):
    pass    
    
###* OWL PROPERTY TYPES 

class FunctionalProperty(ObjectProperty):
    pass
    
class InverseFunctionalProperty(ObjectProperty):
    pass
   
class TransitiveProperty(ObjectProperty):
    pass
    
class SymmetricProperty(ObjectProperty):    
    pass
    
class AsymmetricProperty(ObjectProperty):
    pass
  
class ReflexiveProperty(ObjectProperty):
    pass
   
class IrreflexiveProperty(ObjectProperty):
    pass
    
class AntiSymmetricProperty(ObjectProperty):
    pass
 

class DatatypeProperty(StructuredRel):
    pass
   
    

    
class rdfClass(StructuredNode):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    label_embedding = ArrayProperty(FloatProperty())
    definition = StringProperty()
    definition_embedding = ArrayProperty(FloatProperty())
    subClassOf = RelationshipTo('OntologyClass', 'subClassOf')
    disjointWith = RelationshipTo('OntologyClass', 'disjointWith')
    # unionOf = RelationshipTo('OntologyClass', 'unionOf')
    # intersectionOf = RelationshipTo('OntologyClass', 'intersectionOf')
    
class owlClass(rdfClass):
    pass

## NOTE: instance of owl:Class
class owlThing(StructuredNode):
    pass

class owlNothing(StructuredNode):
    # disjointWith = owlThing
    pass ## NOTE: check how to properly handle this in neomodel
class Individual(StructuredNode):
    uri = StringProperty(unique = True)
    label = StringProperty()
    


from neomodel import StructuredNode, RelationshipTo, StringProperty

class OntologyObjectProperty(StructuredNode):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    definition = StringProperty()
    domain = RelationshipTo('Class', 'DOMAIN')
    range = RelationshipTo('Class', 'RANGE')
    subPropertyOf = RelationshipTo('ObjectProperty', 'subPropertyOf')
    
class FunctionalProperty(OntologyObjectProperty):
    super().__init__()
    pass
class InverseFunctionalProperty(OntologyObjectProperty):
    super().__init__()
    pass
class TransitiveProperty(OntologyObjectProperty):
    super().__init__()
    pass
class SymmetricProperty(OntologyObjectProperty):    
    super().__init__()
    pass
class AsymmetricProperty(OntologyObjectProperty):
    super().__init__()
    pass
class ReflexiveProperty(OntologyObjectProperty):
    super().__init__()
    pass
class IrreflexiveProperty(OntologyObjectProperty):
    super().__init__()
    pass
class AntiSymmetricProperty(OntologyObjectProperty):
    super().__init__()
    pass

    
class OntologyClass(StructuredNode):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    definition = StringProperty()
    subClassOf = RelationshipTo('Class', 'subClassOf')
    disjointWith = RelationshipTo('Class', 'disjointWith')
    


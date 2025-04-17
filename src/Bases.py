from neomodel import StructuredNode, RelationshipTo, StringProperty

class ObjectProperty(StructuredNode):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    definition = StringProperty()
    domain = RelationshipTo('Class', 'DOMAIN')
    range = RelationshipTo('Class', 'RANGE')
    subPropertyOf = RelationshipTo('ObjectProperty', 'subPropertyOf')
    
class Class(StructuredNode):
    uri = StringProperty(unique_index=True)
    label = StringProperty()
    definition = StringProperty()
    subClassOf = RelationshipTo('Class', 'subClassOf')
    disjointWith = RelationshipTo('Class', 'disjointWith')
    


from pyoxigraph import NamedNode, BlankNode

class ObjectProperty:
    def __init__(
        self,
        iri : str
        ):
        self.iri = iri
        self._node = NamedNode(iri)
        
    @property
    def domain(self):
        pass
    
    @property
    def range(self):
        pass


class DatatypePropery:
    def __init__(
        self,
        iri : str
    ):
        self.iri = iri
        

from pyoxigraph import NamedNode

class RDF:
    type = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    Property = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Property")
    XMLLiteral = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#XMLLiteral")
    Statement = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement")
    subject = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#subject")
    predicate = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate")
    object = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#object")
    Bag = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Bag")
    Alt = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Alt")
    Seq = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#Seq")
    List = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#List")
    first = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#first")
    rest = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#rest")
    nil = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#nil")
    langString = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString")
    HTML = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#HTML")
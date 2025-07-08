use petgraph::graph::Graph;
use oxigraph::store::Store;



pub struct OntoClass {
    pub iri: String,
    pub label: String,
    pub definition: String,
}

pub struct OntoProperty {
    pub iri: String,
    pub label: String,
    pub definition : String,
    pub domain: OntoClass,
    pub range : OntoClass,
}



pub struct Ontology{
    pub baseIRI: String,
    pub classes : Vec<OntoClass>,
    pub properties : Vec<OntoProperty>,
    pub hierarchyTree: Graph<N, E>
}



impl Ontology{
    pub fn embed(){

    }
}
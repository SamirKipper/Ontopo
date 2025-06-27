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
use pyo3::prelude::*;
use oxigraph::store::Store;
use oxigraph::io::GraphFormat;
use oxigraph::model::NamedNode;
use oxigraph::sparql::QueryResults;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

mod Structs;
use Structs::*;

/// Your main function exposed to Python or callable from Rust
#[pyfunction]
pub fn run_embedding(file_path: String, base_namespace: String) -> PyResult<()> {
    // Load the ontology file
    let store = Store::new()?;

    // Infer RDF format based on file extension (you can make this smarter later)
    let format = guess_format(&file_path)?;

    let file = File::open(&file_path)?;
    let reader = BufReader::new(file);
    let graph_name = NamedNode::new(&base_namespace)?;

    store.load_graph(reader, format, &graph_name, None)?;

    // Query class info
    let classes = get_class_info(base_namespace, &store)?;

    for class in classes {
        println!("{} - {}", class.iri, class.label);
    }

    Ok(())
}

/// Guess RDF format from file extension (simple heuristic)
fn guess_format(path: &str) -> Result<GraphFormat, PyErr> {
    match Path::new(path).extension().and_then(|s| s.to_str()) {
        Some("ttl") => Ok(GraphFormat::Turtle),
        Some("nt") => Ok(GraphFormat::NTriples),
        Some("rdf") | Some("xml") => Ok(GraphFormat::RdfXml),
        Some("jsonld") => Ok(GraphFormat::JsonLd),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported or unknown RDF format",
        )),
    }
}

/// Extract OWL or RDFS classes from the store
fn get_class_info(base_namespace: String, store: &Store) -> Result<Vec<OntoClass>, Box<dyn std::error::Error>> {
    let query = format!(r#"
        PREFIX base: <{base_namespace}>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>

        SELECT ?class ?label
        WHERE {{
            {{
                ?class rdf:type rdfs:Class ;
                       rdfs:label ?label .
            }}
            UNION
            {{
                ?class rdf:type owl:Class ;
                       rdfs:label ?label .
            }}
        }}
    "#);

    let results = store.query(query)?;
    let mut classes = Vec::new();

    if let QueryResults::Solutions(solutions) = results {
        for solution in solutions {
            let solution = solution?;
            let class = solution
                .get("class")
                .and_then(|t| t.as_named_node())
                .map(|n| n.as_str().to_string());
            let label = solution
                .get("label")
                .and_then(|t| t.as_literal())
                .map(|l| l.value().to_string());

            if let (Some(class), Some(label)) = (class, label) {
                classes.push(OntoClass { iri: class, label });
            }
        }
    }

    Ok(classes)
}

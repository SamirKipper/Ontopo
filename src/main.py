
## * Local
from utils import *
from Bases import *
## * Dependencies
from nltk.corpus import wordnet
import os
from dotenv import load_dotenv
from owlready2 import *
from neomodel import config, db
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.embeddings.openai import OpenAIEmbeddings

def insert_Classes(ontology):
    for c in ontology.classes():
        node = OntologyClass(
            uri = c.iri,
            label = c.label[0],
            definition = c.definition[0],
            )   
        node.save()

# NOTE:
    # Actually this should create a class for intersection of all classes in the is_a list!
def create_hierarchy(ontology):
    for c in ontology.classes():
        for parent in c.is_a:
            if type(parent) != owlready2.entity.ThingClass:
                pass
            else:
                parent_node = OntologyClass.nodes.get(uri = parent.iri)
                child_node = OntologyClass.nodes.get(uri = c.iri) 
                child_node.subClassOf.connect(parent_node)      



if __name__ == "__main__":
    ##* LOAD ENV VARIABLES
    load_dotenv()
    config.DATABASE_URL = os.getenv("NEO4J_ENDPOINT")
    config.NEO4J_USERNAME = os.getenv("NEO4J_USER")
    config.NEO4J_PASSWORD = os.getenv("NEO4j_PASSWORD")
    

    ##* LOAD ONTOLOGIES
    world_2 = World()
    time_1 = world_2.get_ontology("/root/github/EmbedAlign/test/time.rdf").load()
    world_1 = World()
    Ext_R = world_1.get_ontology("/root/github/EmbedAlign/test/extendedRealtionOntology.owx").load() 
    time_bfo = world_1.get_ontology("/root/github/EmbedAlign/test/TimeOntology.rdf").load()
    
    
    ##* CREATE CONNECTION
    #db = Neo4jVector(
    #    os.get_env("NEO4J_ENDPOINT"),
    #    os.get_env("NEO4J_USER"),
    #    os.get_env("NEO4j_PASSWORD"),
    #    OpenAIEmbeddings(),
    #    pre_delete_collection=False,
    #)
    
    
    insert_Classes(time_1)
    create_hierarchy(time_1)
    insert_Classes(time_bfo)
    create_hierarchy(time_bfo)
    
    
    
    ##* START PREPROCESSING
    
    
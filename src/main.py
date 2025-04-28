
## * Local
from utils import *
from Bases import *
## * Dependencies
from nltk.corpus import wordnet
import os
from dotenv import load_dotenv
from owlready2 import *
from neomodel import config
import neo4j
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings






def insert_Classes(ontology, embedder):
    for c in ontology.classes():
        try:
            if len(c.label == 0):
                pass ##! handle issue correctly
            else:
                label = c.label[0] 
                label_embedding = embedder.encode([label])
            if len(c.definition) == 0:
                pass ##! handle issue correctly
            else:
                definition = c.definition[0]
                definition_embedding = embedder.encode([label])
            node = owlClass(
                uri = c.iri,
                label = label,
                label_embedding = label_embedding,
                definition = definition,
                definition_embedding = definition_embedding
                )   
            node.save()
        except Exception as e:
            print(str(e))


def create_hierarchy(ontology):
    for c in ontology.classes():
        for parent in c.is_a:
            if type(parent) != owlready2.entity.ThingClass:
                pass
            else:
                parent_node = owlClass.nodes.get(uri = parent.iri)
                child_node = owlClass.nodes.get(uri = c.iri) 
                child_node.subClassOf.connect(parent_node)      



if __name__ == "__main__":
    
    ##################
    ##* SETUP
    ##################
    
    
    
    ##* LOAD ENV VARIABLES AND SET EMBEDDER
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    load_dotenv()
    config.DATABASE_URL = os.getenv("NEO4J_ENDPOINT")
    config.NEO4J_USERNAME = os.getenv("NEO4J_USER")
    config.NEO4J_PASSWORD = os.getenv("NEO4j_PASSWORD")
    

    ##* LOAD ONTOLOGIES
    world_2 = World()
    time_1 = world_2.get_ontology("/home/kipp_sa/github/EmbedAlign/test/time.rdf").load()
    world_1 = World()
    Ext_R = world_1.get_ontology("/home/kipp_sa/github/EmbedAlign/test/extendedRealtionOntology.owx").load() 
    time_bfo = world_1.get_ontology("/home/kipp_sa/github/EmbedAlign/test/time.rdf").load()
    
    
    ##* CREATE CONNECTION
    db = Neo4jVector(
        url = os.getenv("NEO4J_ENDPOINT"),
        username = os.getenv("NEO4J_USER"),
        password = os.getenv("NEO4j_PASSWORD"),
        embedding = embedder,
        pre_delete_collection=True,     #! FOR NOW
    )
    
    
    
    #######################
    ##* START PREPROCESSING
    #######################
    
    insert_Classes(time_1, embedder)
    create_hierarchy(time_1)
    insert_Classes(time_bfo, embedder)
    create_hierarchy(time_bfo)
    

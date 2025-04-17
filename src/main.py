
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





def load_env():
    load_dotenv()
    config.DATABASE_URL = os.getenv("NEO4J_ENDPOINT")
    config.NEO4J_USERNAME = os.getenv("NEO4J_USER")
    config.NEO4J_PASSWORD = os.getenv("NEO4j_PASSWORD")
    
def load_ontologies(file1, file2):
    try:
        ## NOTE: might require namespace handling
        onto1 = get_ontology(file1).load()
        onto2 = get_ontology(file2).load()
        return onto1, onto2
    except Exception as e:
        return f"Error: {str(e)}"


def align_ontologies(file1: str, file2: str):
    pass


if __name__ == "__main__":
    load_env()
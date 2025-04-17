
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



load_dotenv()



config.DATABASE_URL = os.getenv("NEO4J_ENDPOINT")
config.NEO4J_USERNAME = os.getenv("NEO4J_USER")
config.NEO4J_PASSWORD = os.getenv("NEO4j_PASSWORD")



if __name__ == "__main__":
    pass
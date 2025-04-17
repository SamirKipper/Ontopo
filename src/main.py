import chromadb
from utils import *
from nltk.corpus import wordnet


chroma_client = chromadb
class_label_collection = chroma_client.create_collection(name = "class labels")
class_definition_collection = chroma_client.create_collection(name = "definitions")
property_label_collection = chroma_client.create_collection(name = "property lables")
property_definition_collection = chroma_client.create_collection(name = "definitions")


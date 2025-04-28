- [EmbedAlign](#embedalign)
  - [Installation](#installation)
  - [Running the code](#running-the-code)
  - [useful websites and documentation of dependencies](#useful-websites-and-documentation-of-dependencies)


# EmbedAlign

This Repository explores alignment processes based on vector embedding of the labels and definitions of Ontologies and underlying graphstructure.

## Installation

First, change the password, port and username in the docker compose file to your specific preferences or needs.
Next, create a .env file, that contains the following information:
```env
NEO4J_ENDPOINT="yourPort"
NEO4J_USER="yourUserName"
NEO4j_PASSWORD="YourPassword"
```

Then run the setup.sh with 
```bash
chmod +x setup.sh
```

you should now have a python environment with all necessary dependencies and a neo4j Database running on the port you specified(default: 7474)

>[!NOTE]
>
>Alternatively you can create your own environment and istall from the requirements.txt. The Code uses some downloads from NLTK. For the sake of runtime, it is advisable to download these locally, if you create the environment this way.

## Running the code

## useful websites and documentation of dependencies

- [shapely](https://pypi.org/project/shapely/)
- [neomodel](https://neomodel.readthedocs.io/en/latest/getting_started.html)
- [langchain community Neo4j](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.neo4j_vector.Neo4jVector.html#langchain_community.vectorstores.neo4j_vector.Neo4jVector)
- 
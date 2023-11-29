# OntoView

For interacting with medical ontologies in python


## Installation

Clone the repository and 
```console
pip install -e . 
```

## Usage

To use these ontologies repository, to load up an ontology:

```python
from ontoview.onto import Ontology 
onto = "ontology_concepts.csv"
rels = "ontology_relations.csv"

onto = Ontology(onto, rels)
```

Search the tree structured ontology
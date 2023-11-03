# OntoView

For interacting with medical ontologies in python

[![PyPI - Version](https://img.shields.io/pypi/v/ontoview.svg)](https://pypi.org/project/ontoview)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ontoview.svg)](https://pypi.org/project/ontoview)

-----

**Table of Contents**

- [Installation](#installation)
- [License](#license)

## Installation

```console
pip install ontoview
```

## License

`ontoview` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.


This repository contains the ontologies compatible with the neuroNLP `Ontology` data structure class. Each ontology has been rearranged into the format comparitble with the neuroNLP package; each is made up of two CSV files.
1. ontology_concepts.csv: the list of ontological concepts, including all aliases.
2. ontology_relations.csv: the edge list of ontological relations. Each of the ontologies is structured like a tree, the relations CSV file contains the edges between concepts using their CUI codes.

## Installation

Clone the repository and use the setup command

```
python setup.py install
```

## Usage

To use these ontologies one must first install the neuroOnto repository, to load up an ontology:

```python
from neuroonto.onto import Ontology 
onto = "ontology_concepts.csv"
rels = "ontology_relations.csv"

onto = Ontology(onto, rels)
```

See the neuroNLP repository for full use of the Ontology class.

## Preprocessing
This repository also includes several jupyter notebooks to preprocess ontologies into the neuroOnto custom ontology format.

## Filtering
Several of the contained ontologies are filtered. These are ontologies that have had concepts filtered out according to whether they are present in a text corpus. The `filter_ontology.py` script is used to create these filtered ontologies.
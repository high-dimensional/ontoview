"""Annotator classes for extracting annotations and codes from bioontology

These classes place requests to bioontology to extract SNOMED-CT
and RADLEX codes from texts. See: https://bioportal.bioontology.org/annotator

Classes:
    * OntologyAnnotator
    * SNOMEDCTAnnotator
    * RADLEXAnnotator
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import quote, urlparse
from urllib.request import urlopen

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class OntologyAnnotator:
    """Base text annotator class for bioontology

    This class places URL request to the bioontology webservice
    to annotate clinical texts with codes from standard ontologies.
    """

    def __init__(self):
        self.REST_URL = "http://data.bioontology.org"
        self.API_KEY = "86e65bdb-db27-47d6-a063-9a4ab75f0f7f"

    def _get_json(self, url):
        opener = urllib.request.build_opener()
        opener.addheaders = [("Authorization", "apikey token=" + self.API_KEY)]
        return json.loads(opener.open(url).read())

    def _get_query(self, text):
        return self.REST_URL + "/annotator?text=" + urllib.parse.quote(text)

    def get_annotations(self, texts):
        """Annotate texts with with their ontology codes

        The method uses the api for https://bioportal.bioontology.org/annotator
        webtool to extract clinical ontology codes for texts.

        Parameters
        ----------
        texts : list of strings
            the texts to annotate with ontology codes

        Returns
        -------
        annotated_texts : dict
            a dict with containing the text with a list of character offsets and codes
        """
        # batch_jsons = #map(self.get_json, map(self.get_query, texts))
        # batch_jsons = ((text, self.get_json(self.get_query(text))) for text in texts)
        annotated_texts = []
        for text in tqdm(texts):
            try:
                json_body = self._get_json(self._get_query(text))
                annotation_dict = {"text": text}
                annotation_tuples = []
                for result in json_body:
                    class_details = result["annotatedClass"]
                    # preferred_name = class_details["prefLabel"]
                    id_code_path = class_details["@id"].split("/")
                    id_code = id_code_path[-1]
                    for annotation in result["annotations"]:
                        start = int(annotation["from"]) - 1
                        end = int(annotation["to"])
                    # coded_tuple = (start, end, preferred_name, id_code)
                    coded_tuple = (start, end, id_code)
                    annotation_tuples.append(coded_tuple)
                annotation_dict["annotations"] = annotation_tuples
                annotated_texts.append(annotation_dict)
            except:
                print("Unable to request json")
        return annotated_texts


class SNOMEDCTAnnotator(OntologyAnnotator):
    """The SNOMEDCT annotator using bioontology"""

    def __init__(self):
        super(SNOMEDCTAnnotator, self).__init__()

    def _get_query(self, text):
        return (
            self.REST_URL
            + "/annotator?text="
            + urllib.parse.quote(text)
            + "&ontologies=SNOMEDCT"
        )


class RADLEXAnnotator(OntologyAnnotator):
    """The RADLEX annotator using bioontology"""

    def __init__(self):
        super(RADLEXAnnotator, self).__init__()

    def _get_query(self, text):
        return (
            self.REST_URL
            + "/annotator?text="
            + urllib.parse.quote(text)
            + "&ontologies=RADLEX"
        )

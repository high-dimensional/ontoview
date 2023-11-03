from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.preprocessing import (KBinsDiscretizer, MultiLabelBinarizer,
                                   OneHotEncoder, OrdinalEncoder,
                                   PowerTransformer, StandardScaler)
from umap import UMAP


class BaseVectorizer(BaseEstimator, TransformerMixin):
    """Base vectorizer class

    Parent class for implimentations of spacy doc vectorization
    classes. fit/transform methods are not implemented and this
    class exists only to be inherited by other classes.

    Attributes
    ----------
    pathology_types : list of str
        the pathology types included in the featurization
    section_types : list of str
        the section types included in the featurization
    """

    def __init__(self, pathology_only=False, body_only=True):
        """
        Parameters
        ----------
        body_only : bool, optional
            whether to include the whole report in featurization or just entities in the body
        pathology_only : bool, optional
            whether to use both PATHOLOGY and DESCRIPTOR types in featurization
        """
        super().__init__()
        self.section_types = (
            ["body"]
            if body_only
            else ["body", "header", "indications", "metareport", "tail"]
        )
        self.pathology_types = (
            ["PATHOLOGY"] if pathology_only else ["PATHOLOGY", "DESCRIPTOR"]
        )

    def _are_docs_valid(self, docs):
        """checks if all the docs have the required attributes"""
        return (
            True if all([len(e._.cui) > 1 for doc in docs for e in doc.ents]) else False
        )

    def _get_entities(self, docs):
        """gets all pathology-location entity pairs"""
        entity_pairs = []
        path_filter = lambda e: (e.label_ in self.pathology_types) and (
            not e._.is_negated
        )
        for doc in docs:
            doc_pair_list = []
            for ent in filter(path_filter, doc.ents):
                doc_pair_list.append((ent, None))
                for loc in ent._.relation:
                    doc_pair_list.append((ent, loc))
            entity_pairs.append(doc_pair_list)
        return entity_pairs

    def fit(self, docs, y=None):
        return self

    def transform(self, docs):
        raise NotImplementedError


class OntologyVectorizer(BaseVectorizer):
    """Ontology vectorizer for Doc objects

    Builds a vectorization of a set of docs by
    aligning concepts to nodes in an ontology,
    and enumerating the hierarchy to get all
    concept indices.

    Currently only supports custom ontology.

    Attributes
    ----------
    onto : Ontology object
        the custom ontology object
    include_ancestors : bool, optional
        whether to include the ancestors of a concept in the featurization
    """

    def __init__(
        self,
        ontology,
        include_ancestors=False,
        pathology_only=False,
        max_depth=10,
        include_anatomy=True,
    ):
        """
        Parameters
        ----------
        onto: Ontology object
            the ontology to describe the relations between concepts
        include_ancestors : bool, optional
            whether to include the ancestors of a concept in the featurization
        pathology_only : bool, optional
            whether to use both PATHOLOGY and DESCRIPTOR types in featurization
        """
        super().__init__(pathology_only)
        self.onto = ontology
        self.include_ancestors = include_ancestors
        self._mlb = MultiLabelBinarizer()
        self._ancestor_mapping = {}
        self._root_cui = self.onto.get_root().cui
        self.max_depth = max_depth

    def _find_ancestor_mapping(self, doc_code_pairs):
        all_unique_codes = set(
            [cui for pairs in doc_code_pairs for pair in pairs for cui in pair]
        )
        ancestor_mapping = {}
        for cui in all_unique_codes:
            if cui not in ancestor_mapping.keys():
                current_cui = cui
                cui_ancestors = []
                concept = self.onto[current_cui]
                while concept.parents:
                    if concept.depth <= self.max_depth:
                        cui_ancestors.append(concept.cui)
                    concept = concept.parents[0]
                ancestor_mapping[cui] = cui_ancestors
        return ancestor_mapping

    def _append_ancestors(self, doc_code_pairs):
        new_doc_code_pairs = []
        for pairs in doc_code_pairs:
            old_codes = []  # pairs
            new_codes = [
                (p_ancest, a_ancest)
                for pcui, acui in pairs
                for p_ancest in self._ancestor_mapping[pcui]
                for a_ancest in self._ancestor_mapping[acui]
            ]
            old_codes.extend(new_codes)
            new_doc_code_pairs.append(old_codes)
        return new_doc_code_pairs

    def _get_entity_pair_codes(self, docs, include_ancestors=False):
        if not self._are_docs_valid(docs):
            raise AttributeError(
                "Docs require entity, section, relation, negation and cui attributes to be vectorized"
            )

        entity_pairs = self._get_entities(docs)
        entity_pair_codes = [
            [
                (p._.cui, a._.cui) if a is not None else (p._.cui, self._root_cui)
                for p, a in doc_pairs
            ]
            for doc_pairs in entity_pairs
        ]
        if include_ancestors:
            self._ancestor_mapping = self._find_ancestor_mapping(entity_pair_codes)
            entity_pair_codes = self._append_ancestors(entity_pair_codes)
        return entity_pair_codes

    def fit(self, docs, y=None):
        """fit the model with docs

        Parameters
        ----------
        docs : list of Doc objects
            a series of spacy Docs with entities and relations and negation tagged

        Returns
        -------
        self : object
        """
        entity_pair_codes = self._get_entity_pair_codes(
            docs, include_ancestors=self.include_ancestors
        )
        self._mlb.fit(entity_pair_codes)
        return self

    def transform(self, docs):
        """Apply transformation to docs

        Parameters
        ----------
        docs : list of Doc objects
            a series of spacy Docs with entities relations tagged

        Returns
        -------
        2D numpy binary array
            a numerical array of featurized docs
        """
        entity_pair_codes = self._get_entity_pair_codes(docs)

        if self.include_ancestors:
            entity_pair_codes = self._append_ancestors(entity_pair_codes)

        return self._mlb.transform(entity_pair_codes)

    def fit_transform(self, docs):
        """Fit and apply transformation to docs

        Parameters
        ----------
        docs : list of Doc objects
            a series of spacy Docs with entities relations tagged

        Returns
        -------
        2D numpy binary array
            a numerical array of featurized docs
        """
        entity_pair_codes = self._get_entity_pair_codes(
            docs, include_ancestors=self.include_ancestors
        )

        return self._mlb.fit_transform(entity_pair_codes)

    def inverse_transform(self, doc_array):
        """Return the concept codes contained within doc

        WARNING: input must be binary-valued and of the correct length

        Parameters
        ----------
        doc_array : 2d numpy array
            the binary feature array of a set of vectorized docs

        Returns
        -------
        entity_pairs : list of list of tuples of strings
            a list of entity pair cui codes (pathology, anatomy) for each doc in the input docs
        """
        return self._mlb.inverse_transform(doc_array)

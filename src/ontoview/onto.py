"""Classes for supporting entity linking in the neuroNLP pipeline

The task of entity linking is associating a mention in a text to
an unambiguous concept from a standard ontology. Designed to be used
as utilities alongside the EntityLinker, however the Ontology can
be used separately as an interface to an ontology. These methods are
inspired by the scispacy linking method (https://allenai.github.io/scispacy/).

Classes:
    * Concept
    * Ontology
    * CandidateGenerator
"""
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class Concept:
    """Concept class for entries in an ontology

    Used as a view to a pandas row that holds information
    about a concept from an ontology, including relations
    in the ontological is-a hierarchy.

    Attributes
    ----------
    cui : str
        the concept unique identifier code
    name : str
        the fully-specified (unambiguous) name of the concept
    tui : str
        the type unique identifier code of the concept
    type_name : str
        the name of the concept's type
    parents : list of Concept objects
        a list of the concept's parent Concepts in the is-a hierarchy
    children : list of Concept objects
        a list of the concept's child Concepts in the is-a hierarchy
    is_leaf : bool
        whether the concept is a leaf in the is-a heirarchy
    depth : int
        the depth of the concept in the ontological hierarchy
    """

    def __init__(self, cui, pandas_row, ontology):
        """
        Parameters
        ----------
        pandas_row : row from a pandas DataFrame
            the row from the knowledge base DataFrame containing concept data
        ontology : KnowledgeBase object
            the knowledge base from which the concept is taken
        """
        self.data = pandas_row
        self._cui = cui
        self._onto = ontology

    def __str__(self):
        return "{}, cui: {}".format(str(self.data["concept_name"]), str(self._cui))

    def __repr__(self):
        return self.__str__()

    @property
    def cui(self):
        return self._cui

    @property
    def name(self):
        return self.data["concept_name"]

    @property
    def tui(self):
        return self.data["tui"]

    @property
    def type_name(self):
        return self.data["type_name"]

    @property
    def aliases(self):
        return self._onto.cui2aliases(self._cui)

    @property
    def parents(self):
        parent_ids = self._onto.get_parent_cuis(self._cui)
        return [self._onto[idx] for idx in parent_ids]

    @property
    def children(self):
        child_ids = self._onto.get_child_cuis(self._cui)
        return [self._onto[idx] for idx in child_ids]

    @property
    def is_leaf(self):
        return not bool(len(self.children))

    @property
    def depth(self):
        depth = -1
        node = self
        while node.parents:
            depth += 1
            node = node.parents[0]
        return depth


class Ontology:
    """Ontology interface for a standard ontology

    Contains all the information in a medical ontology.
    Allows access to ontology data and relationships
    between entries.

    Attributes
    ----------
    ontology_name : str
        the name of the ontology
    ontology_dict : pandas DataFrame
        the ontology information held in a pandas DataFrame format
    relation_dict : pandas DataFrame
        the concept relation information for the corresponding ontology
    """

    def __init__(self, ontology_csv_path, relation_csv_path):
        """
        Parameters
        ---------
        ontology_csv_path : str
            the path to the ontology csv data file
        relation_csv_path : str
            the path to the ontology concept relations csv data file
        """

        self.ontology_dict = pd.read_csv(ontology_csv_path, dtype=object)
        self.relation_dict = pd.read_csv(relation_csv_path, dtype=object)
        self.ontology_name = self.ontology_dict["ontology"].unique()[0]
        self._fsm_ontology = self.ontology_dict[
            self.ontology_dict["is_preferred_name"] == "1"
        ].set_index("cui")
        self._fsm_ontology = self._fsm_ontology[
            ~self._fsm_ontology["concept_name"].duplicated()
        ]
        self._alias_ontology = self.ontology_dict[
            self.ontology_dict["is_preferred_name"] == "0"
        ].set_index("cui")
        self._parent_rels = self.relation_dict.set_index("sourceId")
        self._child_rels = self.relation_dict.set_index("destinationId")

    def __len__(self):
        return len(self.ontology_dict)

    def __getitem__(self, code):
        """get a concept from the knowledge base using its cui"""
        if not self.is_in(code):
            raise KeyError("cui {} not present in ontology".format(str(code)))

        row = self._fsm_ontology.loc[code]
        return Concept(code, row, self)

    def is_in(self, code):
        """check if concept is present in the ontology"""
        return code in self._fsm_ontology.index

    def cui2aliases(self, code):
        """get the aliases (names) for the concept using its cui"""
        # rows = self.ontology_dict[
        #    (self.ontology_dict["cui"] == code)
        #    & (self.ontology_dict["is_preferred_name"] == "0")
        # ]
        rows = self._alias_ontology.loc[code]
        alias_names = rows["concept_name"].to_list()
        return alias_names

    def alias2cuis(self, name):
        """find the possible cuis that correspond to a string name"""
        rows = self.ontology_dict[self.ontology_dict["concept_name"] == name.strip()]
        possible_codes = rows["cui"].to_list()
        return possible_codes

    def cui2name(self, code):
        """get the fully-specified name of a concept using its cui"""
        rows = self.ontology_dict[
            (self.ontology_dict["cui"] == code)
            & (self.ontology_dict["is_preferred_name"] == "1")
        ]
        specific_name = rows["concept_name"].values[0]
        return specific_name

    def name2cui(self, name):
        """get the cui of a concept using its fully-specified name"""
        rows = self.ontology_dict[
            (self.ontology_dict["concept_name"] == name.strip())
            & (self.ontology_dict["is_preferred_name"] == "1")
        ]
        if len(rows) == 0:
            raise KeyError("name {} not present in ontology".format(str(name)))
        specific_code = rows["cui"].values[0]
        return specific_code

    def get_parent_cuis(self, code):
        """get the cuis of a concepts parent concepts from its cui"""
        if code not in self._parent_rels.index:
            return []
        parent_ids = self._parent_rels.loc[[code]]["destinationId"].to_list()
        return parent_ids

    def get_child_cuis(self, code):
        """get the cuis of a conceptc child concepts from its cui"""
        if code not in self._child_rels.index:
            return []
        child_ids = self._child_rels.loc[[code]]["sourceId"].to_list()
        return child_ids

    def index2cui(self, index):
        """get the cui of the concept that sits at index in the DataFrame"""
        return self.ontology_dict.iloc[index]["cui"]

    def get_root(self):
        """return the root concept of the ontological hierarchy"""
        all_sources = self._parent_rels.index.unique()
        all_cuis = self._fsm_ontology.index.unique()
        code = all_cuis[~all_cuis.isin(all_sources)].values[0]
        return self.__getitem__(code)

    def validate_ontology(self):
        """check if all concepts in ontology are connected to root"""
        root_node = self.get_root()
        root_cui = root_node.cui
        all_cuis = self.ontology_dict["cui"].unique()
        print("checking parent relations")
        for cui in tqdm(all_cuis):
            path = [cui]
            stack = [cui]
            while stack:
                branch_code = stack.pop(0)
                node = self[branch_code]
                parent_codes = [p.cui for p in node.parents]
                path.extend(parent_codes)
                stack.extend(parent_codes)
            if root_cui not in path:
                print("parent unconnected concept {}".format(cui))
        print("checking child relations")
        stack = [root_cui]
        while stack:
            node_code = stack.pop(0)
            try:
                node = self[node_code]
                stack.extend([c.cui for c in node.children])
            except:
                print("Error trying to access {}".format(node_code))
        print("Ontology validation complete")


class CandidateGenerator:
    """Concept candidate generator

    Given a list of mentions of clinical terms
    in the text, the candidate generator will
    return a list of possible cuis corresponding
    to those mentions, based on tf-idf character
    n-gram similarity.

    Attributes
    ----------
    onto : Ontology object
        the knowledge base ontology from which to draw candidates
    onto_vectors : ndarray
        the tf-idf vectors of the concepts within the knowledgebase
    knn : scikit-learn NearestNeighbours object
        the fitted k-nearest neightbours algorithm object used to find similar tf-idf vectors
    vectorizer : scikit-learn TfidfVectorizer object
        the tf-idf vectorization object that produces the character n-gram vectors.
    """

    def __init__(self, ontology, max_feats=1000, n_grams=(2, 3)):
        """
        Parameters
        ----------
        ontology : Ontology object
            the knowledge base ontology from which to draw candidates
        max_feats : int, optional
            the maximum size of the char 3-gram tf-idf vectors to produce
        n_grams : tuple of int
            the character n-grams to use when constructing character tf-idf vectors
        """
        self.onto = ontology
        self.vectorizer = TfidfVectorizer(
            lowercase=True, analyzer="char", ngram_range=n_grams, max_features=max_feats
        )
        self.onto_vectors = self.vectorizer.fit_transform(
            self.onto.ontology_dict["concept_name"]
        )
        self.knn = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(
            self.onto_vectors
        )
        self.default_cui = self.onto.get_root().cui
        self.cui_encoder = LabelEncoder()
        reference_labels = self.cui_encoder.fit_transform(
            self.onto.ontology_dict["cui"]
        )
        self.knn_cls = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(
            self.onto_vectors, reference_labels
        )

    def get_candidates(self, mention_list, n=5):
        if not mention_list:
            return []
        """return cuis for k-nearest neighour candidate concepts"""
        mention_vectors = self.vectorizer.transform(mention_list)
        distances, indices = self.knn.kneighbors(mention_vectors, n_neighbors=n)
        mention_candidate_cuis = [
            [self.onto.index2cui(idx) for idx in mention_set] for mention_set in indices
        ]
        return mention_candidate_cuis

    def get_zscore_candidate(self, mention_list, zthreshold=-1, n=20):
        """return disambiguated cui by thresholded z score of local similarities"""
        if not mention_list:
            return []
        mention_vectors = self.vectorizer.transform(mention_list)
        distances, indices = self.knn.kneighbors(mention_vectors, n_neighbors=n)
        zscore_dist = zscore(distances, axis=1)
        output_cuis = []
        for i, arg in enumerate(zscore_dist.argmin(axis=1)):
            if zscore_dist[i, arg] < zthreshold:
                output_cuis.append(self.onto.index2cui(indices[i, arg]))
            else:
                output_cuis.append(self.default_cui)
        return output_cuis

    def get_cls_candidate(self, mention_list):
        if not mention_list:
            return []
        mention_vectors = self.vectorizer.transform(mention_list)
        classifications = self.knn_cls.predict(mention_vectors)
        return self.cui_encoder.inverse_transform(classifications)

    def get_best_candidate(self, mention_list, threshold=0.8, n=5):
        """return disambiguated cui by comparing cosine similarities"""
        if not mention_list:
            return []
        mention_vectors = self.vectorizer.transform(mention_list).toarray()
        distances, indices = self.knn.kneighbors(mention_vectors, n_neighbors=n)
        best_candidates = []
        for i, mention_indices in enumerate(indices):
            candidate_vectors = self.onto_vectors[mention_indices].toarray()
            with_mention_vec = np.vstack((mention_vectors[i], candidate_vectors))
            similarity = cosine_similarity(with_mention_vec)[0, 1:]
            if similarity.max() > threshold:
                most_similar_index = mention_indices[similarity.argmax()]
                best_candidate_cui = self.onto.index2cui(most_similar_index)
                best_candidates.append(best_candidate_cui)
            else:
                best_candidates.append(self.default_cui)
        return best_candidates


def get_decendant_cuis(concept):
    cuis = []
    queue = [concept]
    while queue:
        node = queue.pop(0)
        cuis.append(node.cui)
        queue.extend(node.children)
    return cuis


def find_domain_patterns(domain_cuis, ontology):
    """extract the strings associated with concepts
    that are decendents of a set of domain-level concepts"""
    domain_concepts = [ontology[cui] for cui in domain_cuis]
    domain_decendent_cuis = {
        c.cui: get_decendant_cuis(c) for c in tqdm(domain_concepts)
    }
    pathology_domain_dict = {}
    for domain_cui, decendant_cuis in tqdm(domain_decendent_cuis.items()):
        domain_concept_name = ontology[domain_cui].name.lower()
        # domain_string_patterns = [ontology[cui].name.lower() for cui in tqdm(decendant_cuis)]
        domain_rows = ontology.ontology_dict[
            ontology.ontology_dict["cui"].isin(decendant_cuis)
        ]
        domain_string_patterns = domain_rows["concept_name"].str.lower().to_list()
        domain_string_patterns.append(domain_concept_name)
        # domain_alias_patterns = [
        #    alias.lower()
        #    for cui in tqdm(decendant_cuis)
        #    for alias in ontology.cui2aliases(cui)
        # ]
        # domain_string_patterns.extend(domain_alias_patterns)
        pathology_domain_dict[domain_concept_name] = domain_string_patterns

    return pathology_domain_dict

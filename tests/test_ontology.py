import pytest
from neuroNLP.ontology import CandidateGenerator, Concept, Ontology

model_dir = "/home/hwatkins/Desktop/neuroNLP/models"
concepts = model_dir + "/qs_linker_model/ontology_concepts.csv"
relations = model_dir + "/qs_linker_model/ontology_relations.csv"
onto = Ontology(concepts, relations)


def test_concept():
    concept = onto.get_root()
    assert str(concept) == "root_entity, cui: QS7989"
    assert concept.cui == "QS7989"
    assert not concept.is_leaf
    assert concept.depth == -1
    assert len(concept.children) == 3
    assert len(concept.parents) == 0


def test_ontology():
    assert len(onto) == 27690
    assert onto["QS7992"].cui == "QS7992"
    assert not onto.is_in("SDGF")
    assert onto.get_root().name == "root_entity"
    assert onto.get_child_cuis("QS7989") == ["QS7992", "QS7991", "QS7990"]


def test_candidate_generator():
    cgen = CandidateGenerator(onto)
    assert cgen.get_candidates(["brain", "bleed"], n=1) == [["QS6313"], ["QS1078"]]
    print(cgen.get_best_candidate([]))
    assert cgen.get_best_candidate([]) == []
    assert cgen.get_best_candidate(["brain"]) == ["QS6313"]

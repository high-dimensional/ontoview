"""Script to filter an ontology using a text corpus

Only strings that are present in the corpus will be 
included in the final ontology, including ancestors.
"""

import argparse
import os
import string

import pandas as pd
from tqdm import tqdm

from ontoview.onto import CandidateGenerator, Ontology


def main(args):
    input_concepts = args.input_ontology_dir + "/ontology_concepts.csv"
    intput_relations = args.input_ontology_dir + "/ontology_relations.csv"
    onto = Ontology(input_concepts, intput_relations)

    text_corpus_df = pd.read_csv(args.data_path)
    text_corpus = text_corpus_df.report_text.apply(str)

    concept_df = pd.read_csv(input_concepts)
    relations_df = pd.read_csv(intput_relations)

    mega_string = " ".join(text_corpus.sample(args.n_max).to_list()).lower()

    noted_codes = []
    for idx, row in tqdm(concept_df.iterrows(), total=len(concept_df)):
        cui = row.cui
        name = row.concept_name
        if cui not in noted_codes:
            if name in mega_string:
                noted_codes.append(cui)

    additional_codes = []
    for code in tqdm(noted_codes):
        stack = [code]
        while stack:
            branch_code = stack.pop(0)
            node = onto[branch_code]
            parent_codes = [p.cui for p in node.parents]
            additional_codes.extend(parent_codes)
            stack.extend(parent_codes)

    total_codes = set(noted_codes + additional_codes)

    print("Input ontology size {}".format(len(onto)))
    print("Output ontology size {}".format(len(total_codes)))

    filtered_concepts = concept_df[concept_df.cui.isin(total_codes)]
    filtered_relations = relations_df[relations_df.sourceId.isin(filtered_concepts.cui)]

    if os.path.isdir(args.output_ontology):
        raise ValueError("Ontology with that name already exists")
    else:
        os.mkdir(args.output_ontology)

    new_ontology_concepts = args.output_ontology + "/ontology_concepts.csv"
    new_ontology_relations = args.output_ontology + "/ontology_relations.csv"

    filtered_concepts.to_csv(new_ontology_concepts, index=False)
    filtered_relations.to_csv(new_ontology_relations, index=False)

    new_onto = Ontology(new_ontology_concepts, new_ontology_relations)
    new_onto.validate_ontology()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to filter an ontology using a text corpus"
    )
    parser.add_argument("input_ontology_dir", help="Path to the input ontology")
    parser.add_argument("output_ontology", help="Name of output ontology")
    parser.add_argument("data_path", help="Path to csv of text corpus")
    parser.add_argument(
        "-n",
        "--n_max",
        type=int,
        default=1000,
        help="Maximum number of text samples to use",
    )
    args = parser.parse_args()
    main(args)

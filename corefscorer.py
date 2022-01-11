"""Coreference scorer python implementation.

This module provides a python implementation of the most widely used evaluation methods for coreference resolution.
Usage: python3 corefscorer.py --metric <name> --key <file> --response <file> [--sysmentions]

Where the metric name can be MUC, B3, CEAF, BLANC, LEA or simply "all".
Key and response files should be structured as detailed in the readme.md file

Options:
    --help          this message
    --sysmentions   should be enabled when the response set incoudes mentions not in the key set.
"""


import getopt
import sys
import pandas as pd
from tabulate import tabulate

from scorers.B3scorer import B3scorer
from scorers.BLANCscorer import BLANCScorer
from scorers.LEAscorer import LEAScorer
from scorers.Ceafscorer import CEAFScorer
from scorers.mucscore import MUCScorer


def prime_wrapper(df):
    dict_from_csv = {}
    ents = {}

    for index, row in df.iterrows():
        if row['entity_cluster'] in dict_from_csv:
            dict_from_csv[row['entity_cluster']].append(row['mention_id'])
            ents[row['mention_id']] = row['entity_cluster']
        else:
            dict_from_csv[row['entity_cluster']] = [row['mention_id']]
            ents[row['mention_id']] = row['entity_cluster']

    return dict_from_csv, ents


def evaluate(parsed_input, key_ents, parsed_output, response_ents, metric, sysmentions):
    key_set = []
    response_set = []
    for key, value in parsed_input.items():
        key_set.append(set(value))
    for key, value in parsed_output.items():
        response_set.append(set(value))

    if metric == "all":

        Scorer = MUCScorer()
        precision_partition = list(map(lambda p: Scorer.partition_set(p, key_set), response_set))
        recall_partition = list(map(lambda p: Scorer.partition_set(p, response_set), key_set))
        recall_MUC = Scorer.calc_score_muc(response_set, precision_partition)
        precision_MUC = Scorer.calc_score_muc(key_set, recall_partition)
        F1_MUC = 2 * (recall_MUC * precision_MUC) / (recall_MUC + precision_MUC)

        scorer = B3scorer(parsed_output, response_ents, parsed_input, key_ents)
        B3_map = scorer.map_chain()
        recall_B3, precision_B3, F1_B3 = scorer.get_precision_recall_b3(B3_map)

        if sysmentions is True:
            recall_BLANC, precision_BLANC, F1_BLANC = "/", "/", "/"
        else:
            scorer = BLANCScorer(parsed_output, response_ents, parsed_input, key_ents)
            recall_BLANC, precision_BLANC, F1_BLANC = scorer.get_BLANC_score()

        test = LEAScorer(parsed_input, parsed_output)
        test.get_resolution()
        recall_LEA, precision_LEA, F1_LEA = test.LEA_score()

        test = CEAFScorer(parsed_input, parsed_output)
        recall_CEAF, precision_CEAF, F1_CEAF = test.CEAF_score()

        data = [["MUC", recall_MUC, precision_MUC, F1_MUC],
        ["B3", recall_B3, precision_B3, F1_B3],
        ["CEAFe", recall_CEAF, precision_CEAF, F1_CEAF],
        ["BLANC", recall_BLANC, precision_BLANC, F1_BLANC],
        ["LEA", recall_LEA, precision_LEA, F1_LEA]]

        print(tabulate(data, headers=["Metric", "R", "P", "F1"]))

    elif metric == "MUC":
        Scorer = MUCScorer()

        precision_partition = list(map(lambda p: Scorer.partition_set(p, key_set), response_set))
        recall_partition = list(map(lambda p: Scorer.partition_set(p, response_set), key_set))

        recall_MUC = Scorer.calc_score_muc(response_set, precision_partition)
        Precision_MUC = Scorer.calc_score_muc(key_set, recall_partition)

        F1_MUC = 2 * (recall_MUC * Precision_MUC) / (recall_MUC + Precision_MUC)
        print(F1_MUC)

    elif metric == "B3":

        scorer = B3scorer(parsed_output, response_ents, parsed_input, key_ents)
        B3_map = scorer.map_chain()
        print(scorer.get_precision_recall_b3(B3_map))

    elif metric == "CEAF":

        test = CEAFScorer(parsed_input, parsed_output)
        print(test.CEAF_score())

    elif metric == "BLANC":

        scorer = BLANCScorer(parsed_output, response_ents, parsed_input, key_ents)
        print(scorer.get_BLANC_score())

    elif metric == "LEA":

        test = LEAScorer(parsed_input, parsed_output)
        test.get_resolution()
        print(test.LEA_score())


def main():
    sysmention = False

    long_arguments = ["metric=", 'key=', "response=", "sysmentions", "help", ]
    try:
        options, remainder = getopt.getopt(sys.argv[1:], '', long_arguments)
    except getopt.GetoptError:
        print(__doc__)
        return

    for opt, arg in options:
        if opt == '--metric':
            metric = arg
        elif opt == "--key":
            key_file = arg
        elif opt == '--response':
            response_file = arg
        elif opt == '--sysmentions':
            sysmention = True
        elif opt == '--help':
            print(__doc__)
            return

    key_file = pd.read_csv(key_file)
    response_file = pd.read_csv(response_file)

    out_chain, out_ents = prime_wrapper(response_file)
    truth_chain, truth_ents = prime_wrapper(key_file)

    evaluate(truth_chain, truth_ents, out_chain, out_ents, metric, sysmention)


if __name__ == '__main__':
    main()

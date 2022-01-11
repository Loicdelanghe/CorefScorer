import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

class BLANCScorer:
    def __init__(self, output_chains, output_map, truth_chains, truth_map):
        self.output_chains = output_chains
        self.truth_chains = truth_chains
        self.entity_map_key = truth_map
        self.entity_map_response = output_map

    def get_BLANC_score(self):
        rc, wn, wc, rn = self.wrapper()

        precision_coref = rc/(rc + wc)
        precision_non_coref = rn/(rn + wn)
        recall_coref = rc/(rc + wn)
        recall_non_coref = rn/(rn + wc)

        BLANC_P = (precision_coref + precision_non_coref)/2
        BLANC_R = (recall_coref + recall_non_coref)/2

        F_coref = (2*precision_coref*recall_coref)/(precision_coref + recall_coref)
        F_non_coref =(2*precision_non_coref*recall_non_coref)/(precision_non_coref + recall_non_coref)

        BLANC_F = (F_coref + F_non_coref)/2

        return BLANC_P, BLANC_R, BLANC_F





    def wrapper(self):

        extracted_clusters = [value for key, value in self.entity_map_key.items()]
        extracted_clusters_out = [value for key, value in self.entity_map_response.items()]

        yeet_t = list(itertools.combinations(extracted_clusters, 2))
        yeet = list(itertools.combinations(extracted_clusters_out, 2))

        labels_truth = [0 if item1 != item2 else 1 for item1, item2 in yeet_t]
        labels_output = [0 if item1 != item2 else 1 for item1, item2 in yeet]

        conf_matrix = confusion_matrix(labels_truth, labels_output)

        return conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]


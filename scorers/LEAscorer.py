import numpy as np
import pandas as pd
import itertools

class LEAScorer:

    def __init__(self, truth_cluster, system_cluster):

        self.truth_chain = truth_cluster
        self.out_chain = system_cluster
        self.truth_links = {}
        self.out_links = {}

    def get_resolution(self):

        for cluster_id, cluster in self.truth_chain.items():
            if len(cluster) == 1:
                clust_k = list((cluster[0]))
                length_c = 1
                selflink_c = 1
            else:
                clust_k = list(itertools.combinations(cluster, 2))
                length_c = len(cluster)
                selflink_c = length_c * (length_c - 1) / 2

            self.truth_links[cluster_id] = (clust_k, length_c, selflink_c)

        for cluster_id, cluster in self.out_chain.items():
            if len(cluster) == 1:
                clust_k = list((cluster[0]))
                length_c = 1
                selflink_c = 1

            else:
                clust_k = list(itertools.combinations(cluster, 2))
                length_c = len(cluster)
                selflink_c = length_c * (length_c - 1) / 2

            self.out_links[cluster_id] = (clust_k, length_c, selflink_c)

        return self.truth_links, self.out_links

    def LEA_score(self):

        enumerator_r = 0
        denominator_r = 0

        enumerator_p = 0
        denominator_p = 0
        for key, value in self.truth_links.items():
            links_c = value[0]

            selflink_c = value[2]


            length_c = value[1]


            resolution_score_1 = sum([len(set(links_c).intersection(set(links_r[0]))) / selflink_c for key, links_r in self.out_links.items()])


            resolution_score = length_c * (resolution_score_1)


            enumerator_r += resolution_score
            denominator_r += length_c

        for key, value in self.out_links.items():
            links_c = value[0]

            selflink_c = value[2]
            length_c = value[1]

            resolution_score_1 = sum([len(set(links_c).intersection(set(links_r[0]))) / selflink_c for key, links_r in self.truth_links.items()])

            resolution_score = length_c * (resolution_score_1)
            enumerator_p += resolution_score
            denominator_p += length_c

        recall = enumerator_r / denominator_r
        precision = enumerator_p / denominator_p

        F1 = 2 * (recall * precision) / (recall + precision)

        return recall,precision,F1



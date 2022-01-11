from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import itertools

class CEAFScorer:

    def __init__(self,truth_cluster, system_cluster):
        self.truth_chain = truth_cluster
        self.out_chain = system_cluster
        self.truth_cl = list(truth_cluster.values())
        self.system_cl = list(system_cluster.values())
        self.matrix_dim = max(len(self.truth_cl), len(self.system_cl))
        self.empty_mat = np.full((self.matrix_dim, self.matrix_dim), -1)

        self.mappedo = {}
        self.row_matrix_ind = 0
        self.col_matrix_ind = 0

    def get_cost_matrix(self):
        for r in itertools.product(self.truth_cl,self.system_cl):
            if self.col_matrix_ind == len(self.system_cl) - 1:
                self.empty_mat[self.row_matrix_ind,self.col_matrix_ind] = len(list(set(r[0]) & set(r[1])))
                self.mappedo[(self.row_matrix_ind, self.col_matrix_ind)] = (r[0],r[1])

                self.col_matrix_ind = 0
                self.row_matrix_ind += 1

            else:
                self.empty_mat[self.row_matrix_ind, self.col_matrix_ind] = len(list(set(r[0]) & set(r[1])))
                self.mappedo[(self.row_matrix_ind, self.col_matrix_ind)] = (r[0], r[1])
                self.col_matrix_ind += 1

        return self.empty_mat, self.mappedo

    def CEAF_score(self):

        costy, mapped_dict = self.get_cost_matrix()

        row_ind, col_ind = linear_sum_assignment(costy, maximize=True)
        zipped_it = zip(row_ind, col_ind)


        enumerator = 0
        denominator_precision = sum([len(value) for key, value in self.truth_chain.items()])
        denominator_recall = sum([len(value) for key, value in self.out_chain.items()])

        for row, col in zipped_it:
            try:
                one = mapped_dict[(row, col)][0]
                two = mapped_dict[(row, col)][1]
                enumerator += len(list(set(one) & set(two)))

            except KeyError:
                pass

        precision = enumerator / denominator_precision
        recall = enumerator / denominator_recall

        F1 = 2 * (recall * precision) / (recall + precision)

        return precision, recall, F1

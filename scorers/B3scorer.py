import pandas as pd


class B3scorer:
    def __init__(self, output_chains, output_map, truth_chains, truth_map):
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.entity_attributes = {}
        self.output_chains = output_chains
        self.truth_chains = truth_chains
        self.entity_map_key = truth_map
        self.entity_map_response = output_map

    def map_chain(self):
        for cluster_id, entity_ids in self.output_chains.items():
            for entity in entity_ids:
                output_chain_sing = self.output_chains[self.entity_map_response[entity]]
                key_chain = self.truth_chains[self.entity_map_key[entity]]

                truth_chain_l = len(key_chain)
                output_chain_l = len(output_chain_sing)
                output_chain_correct = len(set(entity_ids).intersection(set(key_chain)))
                self.entity_attributes[entity] = (truth_chain_l, output_chain_correct, output_chain_l)

        return self.entity_attributes

    def get_precision_recall_b3(self, maps):
        entity_weight = 1 / len(maps.keys())

        for entity, values in maps.items():
            self.recall += entity_weight * (values[1] / values[0])
            self.precision += entity_weight * (values[1] / values[2])

        self.F1 = 2 * (self.recall*self.precision)/(self.recall+self.precision)

        return self.recall, self.precision, self.F1


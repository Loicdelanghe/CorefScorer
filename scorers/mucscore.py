class MUCScorer:

    def __init__(self):
        self.enumerator = 0
        self.denominator = 0

    def partition_set(self, key, response_set):
        partition_sets = 0
        partition_set = []

        for response in response_set:
            flat_list = [item for sublist in partition_set for item in sublist]
            if key == set(flat_list):
                break

            elif len(key.intersection(response)) >= 1:
                partition_set.append(list(key.intersection(response)))
                partition_sets += 1

        check = [item for sublist in partition_set for item in sublist]
        if len(key) != len(set(check)):
            partition_sets += len(key) - len(set(check))

        return partition_sets

    def calc_score_muc(self, keys, partitions):
        self.enumerator = 0
        self.denominator = 0

        for key in keys:
            ind = keys.index(key)
            partition_set = partitions[ind]

            self.enumerator += (len(key) - partition_set)
            self.denominator += (len(key) - 1)

        try:
            score = self.enumerator / self.denominator
            return score
        except ZeroDivisionError:
            return 0


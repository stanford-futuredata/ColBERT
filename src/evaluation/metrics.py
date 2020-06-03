class Metrics:
    def __init__(self, mrr_depths: dict, recall_depths: dict, total_queries=None):
        self.results = {}
        self.mrr_sums = {depth: 0.0 for depth in mrr_depths}
        self.recall_sums = {depth: 0.0 for depth in recall_depths}
        self.total_queries = total_queries

    def add(self, query_idx, query_key, ranking, gold_positives):
        assert query_key not in self.results
        assert len(self.results) <= query_idx
        assert len(set(gold_positives)) == len(gold_positives)
        assert len(set([pid for _, pid, _ in ranking])) == len(ranking)

        self.results[query_key] = ranking

        positives = [i for i, (_, pid, _) in enumerate(ranking) if pid in gold_positives]

        if len(positives) == 0:
            return

        for depth in self.mrr_sums:
            first_positive = positives[0]
            self.mrr_sums[depth] += (1.0 / (first_positive+1.0)) if first_positive < depth else 0.0

        for depth in self.recall_sums:
            num_positives_up_to_depth = len([pos for pos in positives if pos < depth])
            self.recall_sums[depth] += num_positives_up_to_depth / len(gold_positives)

    def print_metrics(self, query_idx):
        for depth in sorted(self.mrr_sums):
            print("MRR@" + str(depth), "=", self.mrr_sums[depth] / (query_idx+1.0))

        for depth in sorted(self.recall_sums):
            print("Recall@" + str(depth), "=", self.recall_sums[depth] / (query_idx+1.0))


def evaluate_recall(qrels, queries, topK_pids):
    if qrels is None:
        return

    assert set(qrels.keys()) == set(queries.keys())
    recall_at_k = [len(set.intersection(set(qrels[qid]), set(topK_pids[qid]))) / len(qrels[qid]) for qid in qrels]
    recall_at_k = sum(recall_at_k) / len(qrels)
    recall_at_k = round(recall_at_k, 3)
    print("Recall @ maximum depth =", recall_at_k)

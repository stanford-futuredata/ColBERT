import os

def slow_rerank(args, query, pids, passages):
    colbert = args.colbert
    inference = args.inference

    Q = inference.queryFromText([query])

    D_ = inference.docFromText(passages, bsize=args.bsize)
    scores = colbert.score(Q, D_).cpu()

    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    assert len(ranked_pids) == len(set(ranked_pids))

    return list(zip(ranked_scores, ranked_pids, ranked_passages))

import torch


def tensorize_triples(query_tokenizer, doc_tokenizer, queries, passages, scores, bsize, nway):
    # assert len(passages) == len(scores) == bsize * nway
    # assert bsize is None or len(queries) % bsize == 0

    # N = len(queries)
    Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    D_ids, D_mask = doc_tokenizer.tensorize(passages)
    # D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)

    # # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    # maxlens = D_mask.sum(-1).max(0).values

    # # Sort by maxlens
    # indices = maxlens.sort().indices
    # Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    # D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]

    # (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask

    query_batches = _split_into_batches(Q_ids, Q_mask, bsize)
    doc_batches = _split_into_batches(D_ids, D_mask, bsize * nway)
    # positive_batches = _split_into_batches(positive_ids, positive_mask, bsize)
    # negative_batches = _split_into_batches(negative_ids, negative_mask, bsize)

    if len(scores):
        score_batches = _split_into_batches2(scores, bsize * nway)
    else:
        score_batches = [[] for _ in doc_batches]

    batches = []
    for Q, D, S in zip(query_batches, doc_batches, score_batches):
        batches.append((Q, D, S))

    return batches


def _sort_by_length(ids, mask, bsize):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices

    return ids[indices], mask[indices], reverse_indices


def _split_into_batches(ids, mask, bsize):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batches.append((ids[offset:offset+bsize], mask[offset:offset+bsize]))

    return batches


def _split_into_batches2(scores, bsize):
    batches = []
    for offset in range(0, len(scores), bsize):
        batches.append(scores[offset:offset+bsize])

    return batches

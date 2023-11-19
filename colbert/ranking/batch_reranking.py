import os
import time
import torch
import queue
import threading

from collections import defaultdict

from colbert.utils.runs import Run
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, flatten, zipstar
from colbert.indexing.loaders import get_parts
from colbert.ranking.index_part import IndexPart

MAX_DEPTH_LOGGED = 1000  # TODO: Use args.depth


def prepare_ranges(index_path, dim, step, part_range):
    print_message("#> Launching a separate thread to load index parts asynchronously.")
    parts, _, _ = get_parts(index_path)

    positions = [(offset, offset + step) for offset in range(0, len(parts), step)]

    if part_range is not None:
        positions = positions[part_range.start: part_range.stop]

    loaded_parts = queue.Queue(maxsize=2)

    def _loader_thread(index_path, dim, positions):
        for offset, endpos in positions:
            index = IndexPart(index_path, dim=dim, part_range=range(offset, endpos), verbose=True)
            loaded_parts.put(index, block=True)

    thread = threading.Thread(target=_loader_thread, args=(index_path, dim, positions,))
    thread.start()

    return positions, loaded_parts, thread


def score_by_range(positions, loaded_parts, all_query_embeddings, all_query_rankings, all_pids):
    print_message("#> Sorting by PID..")
    all_query_indexes, all_pids = zipstar(all_pids)
    sorting_pids = torch.tensor(all_pids).sort()
    all_query_indexes, all_pids = torch.tensor(all_query_indexes)[sorting_pids.indices], sorting_pids.values

    range_start, range_end = 0, 0

    for offset, endpos in positions:
        print_message(f"#> Fetching parts {offset}--{endpos} from queue..")
        index = loaded_parts.get()

        print_message(f"#> Filtering PIDs to the range {index.pids_range}..")
        range_start = range_start + (all_pids[range_start:] < index.pids_range.start).sum()
        range_end = range_end + (all_pids[range_end:] < index.pids_range.stop).sum()

        pids = all_pids[range_start:range_end]
        query_indexes = all_query_indexes[range_start:range_end]

        print_message(f"#> Got {len(pids)} query--passage pairs in this range.")

        if len(pids) == 0:
            continue

        print_message(f"#> Ranking in batches the pairs #{range_start} through #{range_end}...")
        scores = index.batch_rank(all_query_embeddings, query_indexes, pids, sorted_pids=True)

        for query_index, pid, score in zip(query_indexes.tolist(), pids.tolist(), scores):
            all_query_rankings[0][query_index].append(pid)
            all_query_rankings[1][query_index].append(score)


def batch_rerank(args):
    positions, loaded_parts, thread = prepare_ranges(args.index_path, args.dim, args.step, args.part_range)

    inference = ModelInference(args.colbert, amp=args.amp)
    queries, topK_pids = args.queries, args.topK_pids

    with torch.no_grad():
        queries_in_order = list(queries.values())

        print_message(f"#> Encoding all {len(queries_in_order)} queries in batches...")

        all_query_embeddings = inference.queryFromText(queries_in_order, bsize=512, to_cpu=True)
        all_query_embeddings = all_query_embeddings.to(dtype=torch.float16).permute(0, 2, 1).contiguous()

    for qid in queries:
        """
        Since topK_pids is a defaultdict, make sure each qid *has* actual PID information (even if empty).
        """
        assert qid in topK_pids, qid

    all_pids = flatten([[(query_index, pid) for pid in topK_pids[qid]] for query_index, qid in enumerate(queries)])
    all_query_rankings = [defaultdict(list), defaultdict(list)]

    print_message(f"#> Will process {len(all_pids)} query--document pairs in total.")

    with torch.no_grad():
        score_by_range(positions, loaded_parts, all_query_embeddings, all_query_rankings, all_pids)

    ranking_logger = RankingLogger(Run.path, qrels=None, log_scores=args.log_scores)

    with ranking_logger.context('ranking.tsv', also_save_annotations=False) as rlogger:
        with torch.no_grad():
            for query_index, qid in enumerate(queries):
                if query_index % 1000 == 0:
                    print_message("#> Logging query #{} (qid {}) now...".format(query_index, qid))

                pids = all_query_rankings[0][query_index]
                scores = all_query_rankings[1][query_index]

                K = min(MAX_DEPTH_LOGGED, len(scores))

                if K == 0:
                    continue

                scores_topk = torch.tensor(scores).topk(K, largest=True, sorted=True)

                pids, scores = torch.tensor(pids)[scores_topk.indices].tolist(), scores_topk.values.tolist()

                ranking = [(score, pid, None) for pid, score in zip(pids, scores)]
                assert len(ranking) <= MAX_DEPTH_LOGGED, (len(ranking), MAX_DEPTH_LOGGED)

                rlogger.log(qid, ranking, is_ranked=True, print_positions=[1, 2] if query_index % 100 == 0 else [])

    print('\n\n')
    print(ranking_logger.filename)
    print_message('#> Done.\n')

    thread.join()

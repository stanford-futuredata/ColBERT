import os
import time
import faiss
import random
import torch

from colbert.utils.runs import Run
from multiprocessing import Pool
from colbert.modeling.inference import ModelInference
from colbert.evaluation.ranking_logger import RankingLogger

from colbert.utils.utils import print_message, batch
from colbert.ranking.faiss_index import FaissIndex


def batch_retrieve(args):
    assert args.retrieve_only, "TODO: Combine batch (multi-query) retrieval with batch re-ranking"

    faiss_index = FaissIndex(args.index_path, args.faiss_index_path, args.nprobe, args.part_range)
    inference = ModelInference(args.colbert, amp=args.amp)

    ranking_logger = RankingLogger(Run.path, qrels=None)

    with ranking_logger.context('unordered.tsv', also_save_annotations=False) as rlogger:
        queries = args.queries
        qids_in_order = list(queries.keys())

        for qoffset, qbatch in batch(qids_in_order, 100_000, provide_offset=True):
            qbatch_text = [queries[qid] for qid in qbatch]

            print_message(f"#> Embedding {len(qbatch_text)} queries in parallel...")
            Q = inference.queryFromText(qbatch_text, bsize=512)

            print_message("#> Starting batch retrieval...")
            all_pids = faiss_index.retrieve(args.faiss_depth, Q, verbose=True)

            # Log the PIDs with rank -1 for all
            for query_idx, (qid, ranking) in enumerate(zip(qbatch, all_pids)):
                query_idx = qoffset + query_idx

                if query_idx % 1000 == 0:
                    print_message(f"#> Logging query #{query_idx} (qid {qid}) now...")

                ranking = [(None, pid, None) for pid in ranking]
                rlogger.log(qid, ranking, is_ranked=False)

    print('\n\n')
    print(ranking_logger.filename)
    print("#> Done.")
    print('\n\n')

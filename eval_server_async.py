import asyncio
import aiohttp
import argparse
import sys
import grpc
from concurrent import futures
import torch
import server_pb2
import server_pb2_grpc
from grpc import aio
import psutil
from multiprocessing.connection import Listener
import gc
from colbert.data import Queries
import os.path
import json

import time
import csv

from colbert import Searcher
from collections import defaultdict
import requests
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig

sys.path.append(os.environ["SPLADE_PATH"])

import splade_pb2
import splade_pb2_grpc

class ColBERTServer(server_pb2_grpc.ServerServicer):
    def __init__(self, tag, index, skip_encoding):
        self.tag = tag
        self.suffix = ""
        self.index_name = "wiki.2018.latest" if index == "wiki" else "lifestyle.dev.nbits=2.latest"
        self.multiplier = 250 if index == "wiki" else 500
        self.index_name += self.suffix
        prefix = "/home/ubuntu/data"
        gold_rankings_files = f"{prefix}/{index}/rankings.tsv"
        self.gold_ranks = defaultdict(list)
        self.skip_encoding = skip_encoding

        channel = grpc.aio.insecure_channel('localhost:50060')
        self.splade_stub = splade_pb2_grpc.SpladeStub(channel)

        with open(gold_rankings_files, newline='', encoding='utf-8') as tsvfile:
            tsv_reader = csv.reader(tsvfile, delimiter='\t')
            for line in tsv_reader:
                self.gold_ranks[int(line[0])].append(int(line[1]))
        
        config = Run().config
        config = ColBERTConfig.from_existing(None, config)
        config.load_index_with_mmap = self.suffix != ""
        print(config)
        self.searcher = Searcher(index=f"{prefix}/indexes/{self.index_name}/", config=config)
        self.ranker = self.searcher.ranker
        
        if os.path.isfile(f"{prefix}/{index}/encodings.pt"):
            self.enc = torch.load(f"{prefix}/{index}/encodings.pt")
        elif skip_encoding:
            queries = Queries(path=f"{prefix}/{index}/questions.tsv")
            qvals = list(queries.items())
            self.enc = {}
            for q in qvals:
                self.enc[q[0]] = self.searcher.encode([q[1]])
            torch.save(self.enc, f"{prefix}/{index}/encodings.pt")

    async def fetch_url(self, url, data):
        headers = {'Content-Type': 'application/json'}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=json.dumps(data), headers=headers) as response:
                return await response.text()

    def convert_dict_to_protobuf(self, input_dict):
        query_result = server_pb2.QueryResult()

        query_result.qid = input_dict["qid"]

        for topk_dict in input_dict["topk"]:
            topk_result = query_result.topk.add()
            topk_result.pid = topk_dict["pid"]
            topk_result.rank = topk_dict["rank"]
            topk_result.score = topk_dict["score"]

        return query_result
    
    async def api_serve_query(self, query, qid, k=100):
        gc.collect()
        t2 = time.time()
        url = 'http://localhost:8080'
        splade_q = await self.splade_stub.GenerateQuery(splade_pb2.QueryStr(query=query, multiplier=self.multiplier))
        data = {"query": splade_q.query, "k": 200}
        
        response = await self.fetch_url(url, data)
        response = json.loads(response).get('results', {})
        gr = torch.tensor([int(x) for x in response.keys()], dtype=torch.int)
        Q = self.searcher.encode([query]) if not self.skip_encoding else self.enc[qid]
        scores_, pids_ = self.ranker.score_raw_pids(self.searcher.config, Q, gr)
        print("Searching time of {} on node {}: {}".format(qid, self.tag, time.time() - t2))

        scores_sorter = scores_.sort(descending=True)
        pids_, scores_ = pids_[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
        top_k = []
        for pid, rank, score in zip(pids_, range(len(pids_)), scores_):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})

        del gr
        del scores_
        del pids_
        del Q

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k})

    async def api_search_query(self, query, qid, k=100):
        gc.collect()
        t2 = time.time()
        if not self.skip_encoding:
            pids, ranks, scores = self.searcher.search(query, k)
        else:
            pids, ranks, scores = self.searcher.dense_search(self.enc[qid], k)
        print("Searching time of {}: {}".format(qid, time.time() - t2))

        top_k = []
        for pid, rank, score in zip(pids, ranks, scores):
            top_k.append({'pid': pid, 'rank': rank, 'score': score})
        top_k = list(sorted(top_k, key=lambda p: (-1 * p['score'], p['pid'])))

        del pids
        del ranks
        del scores

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k})

    async def api_rerank_query(self, query, qid, k=100):
        gc.collect()
        t2 = time.time()
        gr = torch.tensor(self.gold_ranks[qid][:200], dtype=torch.int)
        Q = self.searcher.encode([query]) if not self.skip_encoding else self.enc[qid]
        scores_, pids_ = self.ranker.score_raw_pids(self.searcher.config, Q, gr)
        print("Searching time of {} on node {}: {}".format(qid, self.tag, time.time() - t2))

        scores_sorter = scores_.sort(descending=True)
        pids_, scores_ = pids_[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
        top_k = []
        for pid, rank, score in zip(pids_, range(len(pids_)), scores_):
            top_k.append({'pid': pid, 'rank': rank + 1, 'score': score})

        del gr
        del scores_
        del pids_
        del Q

        return self.convert_dict_to_protobuf({"qid": qid, "topk": top_k[:k]})

    async def Search(self, request, context):
        torch.set_num_threads(1)
        return await self.api_search_query(request.query, request.qid, request.k)

    async def Serve(self, request, context):
        torch.set_num_threads(1)
        return await self.api_serve_query(request.query, request.qid, request.k)

    async def Rerank(self, request, context):
        torch.set_num_threads(1)
        return await self.api_rerank_query(request.query, request.qid, request.k)


async def serve_ColBERT_server(args):
    connection = Listener(('localhost', 50040 + psutil.Process().cpu_num()), authkey=b'password').accept()
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_workers))
    server = aio.server()
    server_pb2_grpc.add_ServerServicer_to_server(ColBERTServer(psutil.Process().cpu_num(), args.index, args.skip_encoding), server)
    listen_addr = '[::]:5005' + str(psutil.Process().cpu_num())
    server.add_insecure_port(listen_addr)
    print(f"Starting ColBERT server on {listen_addr}")
    connection.send("Done")
    connection.close()
    await server.start()
    await server.wait_for_termination()
    print("Terminated")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for ColBERT')
    parser.add_argument('-w', '--num_workers', type=int, required=True,
                       help='Number of worker threads per server')
    parser.add_argument('-s', '--skip_encoding', action='store_true',
                        help='Use precomputed encoding')
    parser.add_argument('-i', '--index', type=str, default="search", choices=["wiki", "lifestyle"],
                        required=True, help='Index to run')

    args = parser.parse_args()
    asyncio.run(serve_ColBERT_server(args))

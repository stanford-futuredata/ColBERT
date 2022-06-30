# To Run:
# uvicorn demo-server:app --reload --host 0.0.0.0 --port 4242

# To Query:
# curl -d '{"query":"What is the answer to life, the universe, and everything?"}' -X POST "http://0.0.0.0:4242"

import warnings
warnings.filterwarnings('ignore')

import os
import time
import random
import torch
import itertools

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from types import SimpleNamespace
from collections import OrderedDict

from colbert.utils.runs import Run
from colbert.utils.utils import batch
from colbert.evaluation.loaders import load_colbert
from colbert.indexing.faiss import get_faiss_index_name
from colbert.modeling.inference import ModelInference
from colbert.ranking.rankers import Ranker

random.seed(42)

# Maximum number of passages to return
MAX_QUERY_DEPTH = 10

class Query(BaseModel):
    query: str = ""
    depth: int = 5

app = FastAPI()
origins = [ "*" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO Read config from file
config = SimpleNamespace()
config.lm = "[pretrained_model_name_or_path]"
config.nprobe = 32
config.partitions = 8192
config.faiss_name = None
config.faiss_depth = 1024
config.doc_maxlen = 450
config.mask_punctuation = True
config.bsize = 256
config.amp = True
config.index_root = "./path/to/indexes"
config.root = "./path/to/root"
config.batch = False
config.experiment = "dirty"
config.run = None
config.rank = 0
config.similarity = "cosine"
config.dim = 128
config.query_maxlen = 32
config.qrels = None
config.part_range = None
config.compression_level = 1
config.compression_thresholds = "./compression_thresholds.csv"
config.index_name = "[your_index_name]"
config.checkpoint = "./path/to/colbert.dnn"

config.index_path = os.path.join(config.index_root, config.index_name)
config.faiss_index_path = os.path.join(config.index_path, get_faiss_index_name(config))

# Load ColBERT
config.colbert, config.checkpoint = load_colbert(config)
inference = ModelInference(config.colbert, amp=config.amp)
ranker = Ranker(config, inference, faiss_depth=config.faiss_depth)

# Load passages
passages = {}
passages_tsv = open('./path/to/passages.tsv', 'r')
passages_lines = passages_tsv.readlines()
for line in passages_lines:
    line = line.strip().split("\t")
    passages[int(line[0])] = {
        "text": line[1],
        "title": line[2],
        "link": line[3]
    }

@app.get("/")
async def hello():
    return { "Hello": "Welcome to the demo ColBERT Server!" }

@app.post("/")
async def read_post(query: Query):
    config.run = Run._generate_default_run_name()

    if query.depth < 1:
        query.depth = 1
    if query.depth > MAX_QUERY_DEPTH:
        query.depth = MAX_QUERY_DEPTH

    queries = OrderedDict()
    queries[0] = query.query
    qids_in_order = list(queries.keys())

    # Retrieve!
    retrieveStart = time.time()
    documents = []
    for qoffset, qbatch in batch(qids_in_order, 100, provide_offset=True):
        qbatch_text = [queries[qid] for qid in qbatch]

        rankings = []

        for query_idx, q in enumerate(qbatch_text):
            torch.cuda.synchronize('cuda')

            Q = ranker.encode([q])
            pids, scores = ranker.rank(Q)

            torch.cuda.synchronize()

            rankings.append(zip(pids, scores))

        for query_idx, (qid, ranking) in enumerate(zip(qbatch, rankings)):
            query_idx = qoffset + query_idx

            ranking = [[score, pid, None] for pid, score in itertools.islice(ranking, query.depth)]
            for i in range(query.depth):
                pid = ranking[i][1]
                documents.append(passages[pid])

    retrieveTime = (time.time() - retrieveStart) * 1000.0

    return {
        "results": documents,
        "retrieveTime": retrieveTime
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4242, reload=True, log_level="info")

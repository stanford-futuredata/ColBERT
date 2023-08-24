from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv
from threading import Lock
from time import time

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_ROOT = os.getenv("INDEX_ROOT")
app = Flask(__name__)

searcher = Searcher(index=f"{INDEX_ROOT}/{INDEX_NAME}")
logs = {"requests" : 0, "success": 0, "failure": 0, "latency": 0} 
lock = Lock()

def api_search_query(query, k):
    print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]            
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        lock.acquire()
        logs["requests"] += 1
        lock.release()
        try:
            t1 = time()
            response = api_search_query(request.args.get("query"), request.args.get("k"))
            t2 = time()        
            lock.acquire()
            logs["latency"] = (logs["latency"] * logs["success"] + (t2 - t1)) / (logs["success"] + 1)
            logs["success"] += 1
            lock.release()
            return response
        except e:
            print(e)
            lock.acquire()
            logs["failure"] += 1
            lock.release()
            return ('', 500)
    else:
        return ('', 405)

@app.route("/reset", methods=["GET"])
def reset():
    lock.acquire()
    logs["requests"] = 0
    logs["success"] = 0
    logs["failure"] = 0
    logs["latency"] = 0
    lock.release()
    return logs

@app.route("/logs", methods=["GET"])
def get_logs():
    return logs

if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT")))


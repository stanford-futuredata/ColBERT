import grpc
import asyncio
from concurrent import futures
import psutil
import server_pb2
import server_pb2_grpc

from subprocess import Popen
from threading import Lock
import argparse
import time
import os

from colbert.data import Queries

async def run(nodes):
    print("Process", psutil.Process().cpu_num())
    t = time.time()
    queries = Queries(path="/home/ubuntu/queries.dev.small.tsv")
    qvals = list(queries.items())
    print(qvals)
    tasks = []
    channels = []
    stubs = []

    for i in range(nodes):
        channels.append(grpc.aio.insecure_channel('localhost:5005' + str(i)))
        stubs.append(server_pb2_grpc.ServerStub(channels[-1]))

    for i in range(len(qvals)):
        print(i, channels[i % nodes])
        request = server_pb2.Query(query=qvals[i][1], qid=qvals[i][0], k=100)
        tasks.append(asyncio.ensure_future(stubs[i % nodes].Rerank(request)))

    out = await asyncio.gather(*tasks)

    print(time.time()-t)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ColBERT')
    parser.add_argument('-n', '--num_proc', type=int, required=True,
                        help='Number of servers')
    args = parser.parse_args()

    n = args.num_proc
    processes = []
    for i in range(n):
        processes.append(Popen('PROC_NUM=' + str(i) + " python eval_server.py", shell=True))
        time.sleep(15)
    
    time.sleep(15)
    asyncio.run(run(n))

    for p in processes:
        print("Killing")
        p.kill()

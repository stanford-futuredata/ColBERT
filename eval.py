import grpc
import asyncio
from concurrent import futures
import psutil
import server_pb2
import server_pb2_grpc
import signal
from subprocess import Popen
from threading import Lock
import argparse
import time
import os
from multiprocessing import Pool
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

def start_server(i, t):
    time.sleep(i * t)
    print("Server started in process", psutil.Process().cpu_num())
    return Popen("PROC_NUM=" + str(i) + " taskset -c " + str(psutil.Process().cpu_num()) + \
                 " python eval_server.py", shell=True, preexec_fn=os.setsid).pid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ColBERT')
    parser.add_argument('-n', '--num_proc', type=int, required=True,
                        help='Number of servers')
    parser.add_argument('-t', '--timeout', type=int, default=15,
                        help='Timeout between each process in sec')

    args = parser.parse_args()

    n = args.num_proc
    t = args.timeout
    pool = Pool()
    processes = pool.starmap(start_server, [(i, t) for i in range(n)])
 
    time.sleep(2 * t)
    asyncio.run(run(n))
    pool.terminate()

    for p in processes:
        print("Killing processing after completion")
        os.killpg(os.getpgid(p), signal.SIGTERM)

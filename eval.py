import sys

import grpc
import asyncio
import psutil
import server_pb2
import server_pb2_grpc
import signal
from subprocess import Popen
import argparse
import time
import os
from multiprocessing.connection import Client
from colbert.data import Queries


def save_rankings(rankings, filename):
    output = []
    for q in rankings:
        for result in q.topk:
            output.append("\t".join([str(x) for x in [q.qid, result.pid, result.rank, result.score]]))

    f = open(filename, "w")
    f.write("\n".join(output))
    f.close()


async def run_request(stub, request):
    t = time.time()
    out = await stub.Search(request)
    return out, time.time() - t


async def run(args):
    print("Main process running on CPU", psutil.Process().cpu_num())
    nodes = args.num_proc
    t = time.time()
    queries = Queries(path="/data/queries.dev.small.tsv")
    qvals = list(queries.items())
    print(qvals)
    tasks = []
    channels = []
    stubs = []

    for i in range(nodes):
        channels.append(grpc.aio.insecure_channel('localhost:5005' + str(i)))
        stubs.append(server_pb2_grpc.ServerStub(channels[-1]))

    inter_request_time = [float(x) for x in open(args.input).read().split("\n") if x != ""]
    length = len(inter_request_time)

    for i in range(len(qvals)):
        print(i, channels[i % nodes])
        request = server_pb2.Query(query=qvals[i][1], qid=qvals[i][0], k=100)
        tasks.append(asyncio.ensure_future(run_request(stubs[i % nodes], request)))
        await asyncio.sleep(inter_request_time[i % length])

    await asyncio.sleep(0)
    ret = list(zip(*await asyncio.gather(*tasks)))

    # save_rankings(ret[0], args.output)

    total_time = time.time()-t

    open(args.output, "w").write("\n".join(ret[1]) + f"\nTotal time: {total_time}")
    print(f"Total time for {len(qvals)} requests:",  total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ColBERT')
    parser.add_argument('-n', '--num_proc', type=int, required=True,
                        help='Number of servers')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output file to save results')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file for inter request wait times')

    processes = []
    args = parser.parse_args()

    if args.num_proc > psutil.cpu_count():
        print("Not enough CPUs, exiting!")
        sys.exit(-1)

    for node in range(args.num_proc):
        print("Starting process", node)
        processes.append(Popen("taskset -c " + str(node) + " python eval_server.py",
                               shell=True, preexec_fn=os.setsid).pid)

        times = 10
        for i in range(times):
            try:
                connection = Client(('localhost', 50040 + node), authkey=b'password')
                assert connection.recv() == "Done"
                connection.close()
                break
            except ConnectionRefusedError:
                if i == times - 1:
                    print("Failed to receive connection for child server. Terminating!")
                    for p in processes:
                        os.killpg(os.getpgid(p), signal.SIGTERM)
                    sys.exit(-1)
                time.sleep(5)

    asyncio.run(run(args))

    for p in processes:
        print("Killing processing after completion")
        os.killpg(os.getpgid(p), signal.SIGTERM)

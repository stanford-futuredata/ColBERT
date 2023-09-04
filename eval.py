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


def save_rankings(rankings):
    output = []
    for q in rankings:
        for result in q.topk:
            output.append("\t".join([str(x) for x in [q.qid, result.pid, result.rank, result.score]]))

    f = open("rankings.tsv", "w")
    f.write("\n".join(output))
    f.close()


async def run(nodes):
    print("Process", psutil.Process().cpu_num())
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

    for i in range(len(qvals)):
        print(i, channels[i % nodes])
        request = server_pb2.Query(query=qvals[i][1], qid=qvals[i][0], k=100)
        tasks.append(asyncio.ensure_future(stubs[i % nodes].Rerank(request)))

    await asyncio.sleep(0)
    save_rankings(await asyncio.gather(*tasks))

    print(time.time()-t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluator for ColBERT')
    parser.add_argument('-n', '--num_proc', type=int, required=True,
                        help='Number of servers')

    processes = []
    for cpu in range(psutil.cpu_count()):
        print("Starting process", cpu)
        processes.append(Popen("taskset -c " + str(cpu) + " python eval_server.py",
                               shell=True, preexec_fn=os.setsid).pid)

        connection = Client(('localhost', 50049), authkey=b'password')
        assert connection.recv() == "Done"
        connection.close()

    asyncio.run(run(parser.parse_args().num_proc))

    for p in processes:
        print("Killing processing after completion")
        os.killpg(os.getpgid(p), signal.SIGTERM)

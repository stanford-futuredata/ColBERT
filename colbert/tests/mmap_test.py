import argparse
import copy
import gc
import matplotlib.pyplot
import random
import os, psutil
from multiprocessing import Process, Queue
import sys
import time
import torch

TargetFile = 'mmap_test_data.pt'
TestFile= os.path.join('./colbert/tests', TargetFile)

TEST_FILE_SIZE = 1209306560
TEST_CHUNK_SIZE = 100000
BYTE_SIZE = 1

NUM_COMPUTE_CYCLES = 100

# wait 10 msec to poll
SAMPLE_PERIOD_SEC = 0.01

RESULTS_FILENAME = 'results.png'

B_PER_GB = 1e9

control_basic = []
mmap_basic = []
control_compute = []
mmap_compute = []

def read_into_buffer(mmap, read_buf_size, q):

    print("Starting worker process {}".format(os.getpid()))
    proc = psutil.Process(os.getpid())

    start_time = time.time()

    if mmap:
        storage = torch.ByteStorage.from_file(TestFile, shared=False, size=read_buf_size)

    if mmap:
        mem_buf = torch.ByteTensor(storage)
        mem = proc.memory_info().rss/B_PER_GB
        q.put(mem)
    else:
        mem_buf = torch.load(TestFile, map_location='cpu')
        mem = proc.memory_info().rss/B_PER_GB
        q.put(mem)

    fetch_time = time.time()
    print("it took {}s to read from disk".format(fetch_time - start_time))

    if mmap:
        results_buf = mmap_compute
    else:
        results_buf = control_compute

    copy_avg = 0
    calc_avg = 0
    results_buf = []

    for i in range(NUM_COMPUTE_CYCLES):
        if (i % 10 == 0):
            print("copying cycle {}".format(i))

        rand_time_start = time.time()
        start = random.randrange(0, TEST_FILE_SIZE)
        end = max(start + TEST_CHUNK_SIZE, TEST_FILE_SIZE)
        new_buf = copy.deepcopy(mem_buf[start:end])
        ts1 = time.time()

        computation = torch.sum(new_buf)
        ts2 = time.time()

        copy_avg = copy_avg + ts1 - rand_time_start
        calc_avg = calc_avg + ts2 - ts1

        del new_buf
        gc.collect()

        mem = proc.memory_info().rss
        results_buf.append(mem/B_PER_GB)

    copy_avg = copy_avg/NUM_COMPUTE_CYCLES
    calc_avg = calc_avg/NUM_COMPUTE_CYCLES
    print("copy average time = {}, calculation average time = {}".format(copy_avg, calc_avg))

    del mem_buf
    gc.collect()

    end_time = time.time()
    print("fetch to end time = {}".format(end_time - fetch_time))

    q.put(results_buf)

    return


# run_test(mmap: bool)
#   mmap -          if true, use memory mapping to save files to disk
#                   else, save to disk and load entire files into memory
#   test_data -     contents of file to write
#   test_iter -     how many test cycles to run
#   read_buf_size - how much data to read into memory
#
#   results -       list of length test_iter with memory usage
#
# This test writes a file to disk, and then spawns a worker process
# to fill up a buffer in memory test_iter times. The size of the
# in-memory buffer is given by read_buf_size.
#
# The parent process will monitor the memory usage of the worker
# process for each test iteration and report back the results to
# the caller for analysis.
def run_test(mmap, test_iter, read_buf_size):
    for i in range(test_iter):
        print("Starting iteration {}".format(i))

        q = Queue()

        start_time = time.time()
        worker_proc = Process(target=read_into_buffer, args=(mmap, read_buf_size, q))
        worker_proc.start()

        worker_proc.join()
        end_time = time.time()

        print("exec time = {}".format(end_time - start_time))

        if mmap:
            mmap_basic.append(q.get())
            mmap_compute.extend(q.get())
        else:
            control_basic.append(q.get())
            control_compute.extend(q.get())

def main(args):
    target_dir = args.target_dir

    test_iter = args.test_iter
    read_buf_size = args.read_buf_size

    ### Start the test
    print("\nStarting MMap stress test")

    ### Read random parts of the test data file with and without MMap
    print("\nStarting random file read control test")
    run_test(False, test_iter, read_buf_size)

    print("\nStarting random file read mmap test")
    run_test(True, test_iter, read_buf_size)

    ### Cleanup
    print("\nFinished MMap stress test\n")

    ### Results
    # print table and generate graph with results of memory mapping the
    #   files and not, in terms of memory pressure

    print("Saving Results\n--------------")
    results = [control_basic, mmap_basic, control_compute, mmap_compute]
    labels = ['control_basic', 'mmap_basic', 'control_compute', 'mmap_compute']

    fig = matplotlib.pyplot.figure()
    plot = fig.add_subplot(111)

    plot.boxplot(results, labels=labels, showmeans=True)
    plot.set_ylabel('Memory Usage (GB)')

    result_filename = os.path.join(target_dir, RESULTS_FILENAME)
    matplotlib.pyplot.savefig(result_filename)

    print("Results saved to {}".format(result_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MMap stress test")

    # required arguments
    parser.add_argument(
        "--target_dir", type=str, required=True, help="Path to directory to save files to"
    )

    # optional arguments
    parser.add_argument(
        "--test_iter", type=int, required=False, default=1, help="Number of test cycles"
    )
    parser.add_argument(
        "--read_buf_size", type=int, required=False, default=TEST_FILE_SIZE,\
        help="Number of copies of the test data to read into memory"
    )

    args = parser.parse_args()
    main(args)

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

TargetFile = 'ivf.pt'
TestFile = os.path.join('./colbert/tests', TargetFile)
HDD_filepath = os.path.join('/lfs/1/udingank/msmarco.nbits-2.latest', TargetFile)
SSD_filepath = os.path.join('/lfs/0/udingank/msmarco.nbits-2.latest', TargetFile)
Sources = [HDD_filepath, SSD_filepath]

Results = ['results-hdd.png', 'results-ssd.png']

verbose = False

TEST_FILE_SIZE = 9566559787 
TEST_CHUNK_SIZE = 100000
BYTE_SIZE = 1

NUM_COMPUTE_CYCLES = 1000

B_PER_GB = 1e9

control_basic = []
mmap_basic = []
control_compute = []
mmap_compute = []

control_timing = []
mmap_timing = []

def read_into_buffer(mmap, read_buf_size, filepath, q):

    if verbose:
        print("Starting worker process {}".format(os.getpid()))

    # read time, average copy time, average calculation time, total time
    timing = { 'read_time': 0, 'average_copy_time': 0, 'average_calc_time': 0, 'total_time': 0 }
    proc = psutil.Process(os.getpid())

    start_time = time.time()

    if mmap:
        storage = torch.ByteStorage.from_file(filepath, shared=False, size=read_buf_size)

    if mmap:
        mem_buf = torch.ByteTensor(storage)
        mem = proc.memory_info().rss/B_PER_GB
        q.put(mem)
    else:
        mem_buf = torch.load(filepath, map_location='cpu')[0]
        mem = proc.memory_info().rss/B_PER_GB
        q.put(mem)

    fetch_time = time.time()
    timing['read_time'] = fetch_time - start_time
    if verbose:
        print("it took {}s to read from disk".format(timing['read_time']))

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

        copy_avg += ts1 - rand_time_start
        calc_avg += ts2 - ts1

        mem = proc.memory_info().rss
        del new_buf
        gc.collect()

        results_buf.append(mem/B_PER_GB)

    copy_avg = copy_avg/NUM_COMPUTE_CYCLES
    calc_avg = calc_avg/NUM_COMPUTE_CYCLES
    if verbose:
        print("copy average time = {}, calculation average time = {}".format(copy_avg, calc_avg))
    timing['average_copy_time'] = copy_avg
    timing['average_calc_time'] = calc_avg

    del mem_buf
    gc.collect()

    end_time = time.time()
    timing['total_time'] = end_time - fetch_time
    if verbose:
        print("fetch to end time = {}".format(timing['total_time']))

    q.put(results_buf)
    q.put(timing)

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
def run_test(mmap, test_iter, read_buf_size, filepath):

    for i in range(test_iter):
        print("Starting iteration {}".format(i))

        q = Queue()

        start_time = time.time()
        worker_proc = Process(target=read_into_buffer, args=(mmap, read_buf_size, filepath, q))
        worker_proc.start()

        worker_proc.join()
        end_time = time.time()

        print("exec time = {}".format(end_time - start_time))

        if mmap:
            mmap_basic.append(q.get())
            mmap_compute.extend(q.get())
            mmap_timing.append(q.get())
        else:
            control_basic.append(q.get())
            control_compute.extend(q.get())
            control_timing.append(q.get())

def print_results(target_dir, filepath):
    # print table and generate graph with results of memory mapping the
    #   files and not, in terms of memory pressure

    print("Saving Results\n--------------")
    results = [control_basic, mmap_basic, control_compute, mmap_compute]

    # create labels
    labels = ['control_basic', 'mmap_basic', 'control_compute', 'mmap_compute']

    # create basic figure
    fig = matplotlib.pyplot.figure()
    plot = fig.add_subplot(111)

    # plot the box plot
    bp = plot.boxplot(results, labels=labels, showmeans=True)
    plot.set_ylabel('Memory Usage (GB)')

    result_filename = os.path.join(target_dir, filepath)
    matplotlib.pyplot.savefig(result_filename)

    print("Results saved to {}\n".format(result_filename))

    for i, l in enumerate(results):
        print("{:10s}: {:.2f} GB".format(labels[i], sum(l)/len(l)))

def print_timing(test_iter):
    print("\nTiming Summary")
    print("--------------\n")

    control = { 'read_time': 0, 'average_copy_time': 0, 'average_calc_time': 0, 'total_time': 0 }
    mmap = { 'read_time': 0, 'average_copy_time': 0, 'average_calc_time': 0, 'total_time': 0 }

    for i in range(test_iter):
        control_iter = control_timing[i]
        for k in control_iter:
            control[k] += control_iter[k]

    for k in control:
        control[k] /= test_iter

    for i in range(test_iter):
        mmap_iter = mmap_timing[i]
        for k in mmap_iter:
            mmap[k] += mmap_iter[k]

    for k in mmap:
        mmap[k] /= test_iter

    print("  Timing   |  Control  |  MMap  ")
    print("-----------|-----------|--------------")
    print(" read time |    {:.2f}   |    {:.2f}     ".format(control['read_time'], mmap['read_time']))
    print(" copy time |    {:.2f}   |    {:.2f}     ".format(control['average_copy_time'], mmap['average_copy_time']))
    print("  compute  |    {:.2f}   |    {:.2f}     ".format(control['average_calc_time'], mmap['average_calc_time']))
    print("   total   |    {:.2f}   |    {:.2f}     ".format(control['total_time'], mmap['total_time']))


def main(args):
    target_dir = args.target_dir

    test_iter = args.test_iter
    read_buf_size = args.read_buf_size

    ### Start the test
    print("\nStarting MMap stress test")

    ### Read random parts of the test data file with and without MMap
    for i, filepath in enumerate(Sources):
        print("\nStarting random file read control test from {}".format(filepath))
        run_test(False, test_iter, read_buf_size, filepath)

        print("\nStarting random file read mmap test from {}".format(filepath))
        run_test(True, test_iter, read_buf_size, filepath)

        ### Results
        print_results(target_dir, Results[i])

        ### Timing Summary
        print_timing(test_iter)

        control_basic = []
        mmap_basic = []
        control_compute = []
        mmap_compute = []

        control_timing = []
        mmap_timing = []

    print("\nFinished MMap stress test\n")


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

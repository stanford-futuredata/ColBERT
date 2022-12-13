import argparse
import os, psutil
from multiprocessing import Process
import time
import torch

TargetFile = 'mmap_test_data.pt'
TestFile= os.path.join('./colbert/tests', TargetFile)

TEST_FILE_SIZE = 120930656
BYTE_SIZE = 1

NUM_READ_CYCLES = 10

# wait 10 msec to poll
SAMPLE_PERIOD_SEC = 0.01


def write_mmapped(target_dir):
    test_data = torch.load(TestFile, map_location='cpu')

    # write using mmap API
    storage = torch.ByteStorage.from_file(filename=os.path.join(target_dir, TargetFile), shared=True, size=TEST_FILE_SIZE)
    torch.ByteTensor(storage).copy_(torch.flatten(test_data))


def write_to_disk(target_dir):
    test_data = torch.load(TestFile, map_location='cpu')

    # write to disk not using mmap
    torch.save(test_data, os.path.join(target_dir, TargetFile))


def read_into_buffer(mmap, read_buf_size, target_dir):
    # create memory buffer, read into it NUM_READ_CYCLES times
    print("Starting worker process {}".format(os.getpid()))

    target_path = os.path.join(target_dir, TargetFile)

    mem_buf = torch.empty(read_buf_size, BYTE_SIZE, dtype=torch.uint8)

    if mmap:
        storage = torch.ByteStorage.from_file(filename=target_path, shared=False, size=read_buf_size)

    for i in range(NUM_READ_CYCLES):
        # read into buffer
        if mmap:
            mem_buf = torch.ByteTensor(storage)
        else:
            mem_buf = torch.load(target_path, map_location='cpu')

    return


# run_test(mmap: bool)
#   mmap -          if true, use memory mapping to save files to disk
#                   else, save to disk and load entire files into memory
#   target_dir -    where to write the file to
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
def run_test(mmap, target_dir, test_iter, read_buf_size):

    results = []

    for i in range(test_iter):
        print("Starting iteration {}".format(i))
        # Write test data to disk using mmap or not
        if mmap:
            write_mmapped(target_dir)
        else:
            write_to_disk(target_dir)

        # Spawn a process and track the worker's PID
        worker_proc = Process(target=read_into_buffer, args=(mmap, read_buf_size, target_dir))
        worker_proc.start()
        worker_pid = worker_proc.pid

        # While the worker is not finished, periodically track memory usage
        cur_result = []
        worker_mem_proc = psutil.Process(worker_pid)
        while worker_proc.is_alive():
            #print("memory info: {}".format(worker_mem_proc.memory_info()))
            mem = worker_mem_proc.memory_info().rss
            cur_result.append(mem)
            time.sleep(SAMPLE_PERIOD_SEC)

        worker_proc.join()

        result = sum(cur_result)/len(cur_result)
        results.append(result)
        print("iter {} = {}".format(i, result))

    return results


def main(args):
    target_dir = args.target_dir

    test_iter = args.test_iter
    read_buf_size = args.read_buf_size

    # Start the test
    print("\nStarting MMap stress test")

    # Run the test without MMap, save memory usage stats
    print("\nStarting control test")
    control_results = run_test(False, target_dir, test_iter, read_buf_size)

    # Run the test again with MMap, save memory usage stats
    print("\nStarting mmap test")
    mmap_results = run_test(True, target_dir, test_iter, read_buf_size)

    # Cleanup
    print("\nFinished MMap stress test\n")

    # Results
    # print table and generate graph with results of memory mapping the
    #   files and not, in terms of memory pressure

    print("Results\n-------")
    print("Control Test:\niter    memory usage")
    for i in range(len(control_results)):
        print("{} {}".format(i, control_results[i]))

    print("MMap Test:\niter    memory usage")
    for i in range(len(mmap_results)):
        print("{} {}".format(i, mmap_results[i]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MMap stress test")

    # required arguments
    parser.add_argument(
        "--target_dir", type=str, required=True, help="Path to directory to save files to"
    )

    # optional arguments
    parser.add_argument(
        "--test_iter", type=int, required=False, default=3, help="Number of test cycles"
    )
    parser.add_argument(
        "--read_buf_size", type=int, required=False, default=100,\
        help="Number of copies of the test data to read into memory"
    )

    args = parser.parse_args()
    main(args)

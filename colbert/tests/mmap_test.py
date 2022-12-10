import os
import argparse
from multiprocessing import Process

# <TODO> : populate this file!
TestData = os.path.join('./', 'mmap_test_data.pt')
NUM_READ_CYCLES = 10
SAMPLE_PERIOD_MSEC = 10


# <TODO>
def write_mmapped(target_dir, test_data):
    # write using mmap API
    pass


# <TODO>
def write_to_disk(target_dir, test_data):
    # write to disk not using mmap
    pass


# <TODO>
def read_into_buffer(read_buf_size, target_dir):
    # create memory buffer, read into it NUM_READ_CYCLES times
    print("Starting worker process {}".format(os.getpid()))
    pass


# run_test(mmap: bool) <TODO>
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

    for _ in xrange(test_iter):
        # Write test data to disk using mmap or not
        if mmap:
            write_mmapped(target_dir)
        else:
            write_to_disk(target_dir)

        # Spawn a process and track the worker's PID
        worker_proc = Process(target=read_into_buffer, args=(read_buf_size, target_dir))
        worker_pid = worker_proc.pid
        worker_proc.start()

        # While the worker is not finished, periodically track memory usage
        cur_result = []
        while worker_proc.is_alive():
            # <TODO>: get memory usage, save, wait SAMPLE_PERIOD_MSEC
            continue

        worker_proc.join()

        results.append(sum(cur_result)/len(cur_result))

    return results


def main(args):
    target_dir = args.target_dir

    test_iter = args.test_iter
    read_buf_size = args.read_buf_size

    # Start the test
    print("Starting MMap stress test")

    # Run the test without MMap, save memory usage stats
    control_results = run_test(mamp=False, target_dir, test_iter, read_buf_size)

    # Run the test again with MMap, save memory usage stats
    mmap_results = run_test(mmap=True, target_dir, test_iter, read_buf_size)

    # Cleanup
    print("Finished MMap stress test")

    # Results
    # <TODO>
    # print table and generate graph with results of memory mapping the
    #   files and not, in terms of memory pressure

    print("Results:")


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
        "--read_buf_size", type=int, required=False, default=10,\
        help="Number of copies of the test data to read into memory"
    )

    args = parser.parse_args()
    main(args)

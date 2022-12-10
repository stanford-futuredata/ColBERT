import os
import argparse


TestData = os.path.join('./', 'mmap_test_data.pt')

# run_test(mmap: bool) <TODO>
#   mmap -          if true, use memory mapping to save files to disk
#                   else, save to disk and load entire files into memory
#   target_dir -    where to write the file to
#   test_data -     contents of file to write
#   test_iter -     how many test cycles to run
#   read_buf_size - how much data to read into memory
#
#   results -       list of memory usage over test duration
#
# This test writes a file to disk, and then spawns a worker thread
# to fill up a buffer in memory test_iter times. The size of the
# in-memory buffer is given by read_buf_size.
#
# The parent thread will monitor the memory usage of the worker
# thread for each test iteration and report back the results to
# the caller for analysis.
def run_test(mmap):
    # Write test data to disk using the from_file API
    # Spawn a thread and track the worker's PID
    # While the worker is not finished, track the number
    return results


def main(args):
    target_dir = args.target_dir

    test_iter = args.test_iter
    read_buf_size = args.read_buffer_size

    # Start the test
    print("Starting MMap stress test")

    # Run the test without MMap, save memory usage stats
    control_results = run_test(mamp=False, target_dir, test_data, test_iter, read_buf_size)

    # Run the test again with MMap, save memory usage stats
    mmap_results = run_test(mmap=True, target_dir, test_data, test_iter, read_buf_size)

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
        "--read_buffer_size", type=int, required=False, default=10,\
        help="Number of copies of the test data to read into memory"
    )

    args = parser.parse_args()
    main(args)

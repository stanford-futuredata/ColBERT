import argparse
import math
import random

def main(args):
    inter_arrival_time_generator = random.Random()
    inter_arrival_time_generator.seed(args.seed + 1)

    def generate_inter_arrival_time(rng, lam):
        return -math.log(1.0 - rng.random()) * lam

    tot = 0
    with open(args.output_file, 'w') as f:
        while True:
            inter_arrival_time = generate_inter_arrival_time(inter_arrival_time_generator, args.lam)
            if tot < args.num_requests:
                f.write('%f\n' % inter_arrival_time)
                tot += 1
            else:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic trace')
    parser.add_argument('-n', '--num_requests', type=int, required=True,
                        help='Number sec to generate')
    parser.add_argument('-l', '--lam', type=float, default=0.0,
                        help='Lambda for Poisson arrival rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('-o', '--output_file', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()
    main(args)

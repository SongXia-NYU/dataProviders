import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--pattern", type=str, default="job_{}.sbatch")
    args = parser.parse_args()
    for i in range(args.start, args.end):
        os.system(f"sbatch {args.pattern.format(i)} ")

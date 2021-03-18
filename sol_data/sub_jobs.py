from glob import glob
import os

if __name__ == '__main__':
    files = glob("raw/jobs/job*.pbs")
    for f in files:
        os.system("sbatch {}".format(f))

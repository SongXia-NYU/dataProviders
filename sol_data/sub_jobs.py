from glob import glob
import os

if __name__ == '__main__':
    start = 0
    end = 641
    files = "raw/freesolv_oct_logs/job_{}.sbatch"
    for i in range(start, end):
        f_name = files.format(i)
        os.system("sbatch {}".format(f_name))

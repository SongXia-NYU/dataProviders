import argparse
from glob import glob
import os.path as osp
import os


def main(input_dir, output_dir, time, mem, cpus, tasks_per_job, job_pattern):
    os.makedirs(output_dir, exist_ok=True)
    input_com_files = glob(osp.join(input_dir, "*.com"))
    n_com_files = len(input_com_files)
    starts = list(range(0, n_com_files, tasks_per_job))
    for job_id, start in enumerate(starts):
        end = start + tasks_per_job
        if end > n_com_files:
            end = n_com_files
        with open(osp.join(output_dir, job_pattern.format(job_id)), "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("#SBATCH --job-name=G{}\n".format(job_id))
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --ntasks-per-node=1\n")
            f.write("#SBATCH --cpus-per-task={}\n".format(cpus))
            f.write("#SBATCH --time={}:00:00\n".format(time))
            f.write("#SBATCH --mem={}GB\n\n".format(mem))
            f.write("module purge\n")
            f.write("module load gaussian/intel/g16a03\n\n")
            for i in range(start, end):
                basename = osp.basename(input_com_files[i])
                output_file = osp.join(output_dir, basename.split(".")[0] + ".log")
                f.write("run-gaussian {} > {}\n".format(osp.abspath(input_com_files[i]), osp.abspath(output_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--job_pattern", type=str, default="job_{}.sbatch")
    parser.add_argument("--tasks_per_job", type=int, default=2)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, time=10, mem=3, cpus=2, tasks_per_job=args.tasks_per_job,
         job_pattern=args.job_pattern)

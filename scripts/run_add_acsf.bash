#!/bin/bash
#
#SBATCH --job-name=vis_al
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=40GB

module purge
singularity exec --nv --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1-extra-5GB-3.2M.ext3:ro --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1.sqf:ro \
       /scratch/work/public/hudson/images/rocm-4.0.1.sif bash -c \
       "source /ext3/env.sh; export PYTHONPATH=..:$PYTHONPATH; python add_acsf.py --dataset_name frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-MMFF.pt
       --save_name frag20reducedAllSolRef-Bmsg-cutoff-10.00-sorted-defined_edge-lr-acsf308-MMFF.pt"

module purge
singularity exec --nv --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1-extra-5GB-3.2M.ext3:ro --overlay ~/conda_envs/pytorch1.8.1-rocm4.0.1.sqf:ro \
       /scratch/work/public/hudson/images/rocm-4.0.1.sif bash -c \
       "source /ext3/env.sh; export PYTHONPATH=..:$PYTHONPATH; python add_acsf.py"	

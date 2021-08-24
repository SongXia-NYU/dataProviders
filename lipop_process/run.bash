singularity exec --nv --overlay ../sol_data/lipop-25GB-500K.ext3 --overlay /scratch/sx801/singularity-envs/pytorch1.7.0-cuda11.0-extra-5GB-3.2M.ext3:ro --overlay ~/conda_envs/dnn-3.7-pytorch1.7.0-cuda11.0.ext3:ro /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

export PYTHONPATH=/scratch/sx801/scripts/physnet-dimenet/dataProviders:../../Design-16-uncertainty/


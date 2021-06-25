module purge
singularity exec --nv \
--overlay ~/conda_envs/dnn-3.7-pytorch1.7.0-cuda11.0.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-10-gas1-50G-10M.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-15-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-11-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-14-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-20-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-9-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-16-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-13-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-10-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-10-gas2-50G-10M.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-12-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-9-gas1-50G-10M.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-17-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-18-25GB-500K.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-9-gas2-50G-10M.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-10-gas3-50G-10M.ext3:ro \
--overlay /archive/sx801/data/Frag20-Sol/Frag20-Sol-log/frag20-19-25GB-500K.ext3:ro \
/scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif /bin/bash

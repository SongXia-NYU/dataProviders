#!/bin/bash -e

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:10:00
#SBATCH --mem=10GB
#SBATCH --job-name=sol
#SBATCH --array=0-1600

module purge
module load gaussian/intel/g16a03

input_list=plati20_gas_round1.txt 

start=$((SLURM_ARRAY_TASK_ID*14))
end=$((start+14))

i=0
while read -r line; do
    if [[ $i -ge $start ]] && [[ $i -lt $end ]]; then
	input=$line
	output=$(echo $input | sed -e s'/.com$/.log/')
	timeout 5m run-gaussian $input $output
    fi
    i=$((i+1))
done < $input_list



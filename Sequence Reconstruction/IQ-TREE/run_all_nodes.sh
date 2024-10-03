#!/bin/bash

########## SBATCH Lines for Resource Request ##########

#SBATCH --time=60:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1              # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1              # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=4    # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=40G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name sh    # you can give your job a name for easier identification (same as -J)


module load   GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
eval "$(conda shell.bash hook)"
conda activate /mnt/home/mardikor/anaconda3/
conda activate mardikor_DL

# The path to the input and output files
input_file="ASR.txt"
output_file="training_PK2_sh_ultra_1000.fasta"
NUM_SEQUENCES=1000
gap_info_file="Binary.csv"
file="ultra_sh.txt"

python seq_from_state.py $input_file $output_file $NUM_SEQUENCES  $file  $gap_info_file  


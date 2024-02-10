#!/bin/bash
#SBATCH --job-name=run
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
##SBATCH --time=3-00:00            # Runtime in D-HH:MM
#SBATCH --partition=bethge
##SBATCH --partition=gpu-2080ti    # Partition to submit to
##SBATCH --partition=gpu-2080ti-preemptable    # Partition to submit to
#SBATCH --output=slurm/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=slurm/%x_%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=10000

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

if [ $1 == 0 ]; then
    log=/mnt/qb/work/bethge/ysharma/projects/concept_graphs/logs/synthetic
    echo "synthetic";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/cuda_11.3.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/concept_graphs/train.py --debug 2>&1 | tee ${log}.txt"
elif [ $1 == 1 ]; then
    log=/mnt/qb/work/bethge/ysharma/projects/concept_graphs/logs/celeba
    echo "celeba";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/cuda_11.3.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/concept_graphs/train.py --debug --dataset celeba-3classes-10000 2>&1 | tee ${log}.txt"
fi
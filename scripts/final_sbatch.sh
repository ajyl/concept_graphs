#!/bin/bash
#SBATCH --job-name=final
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --nodes=1                 # Ensure that all cores are on one machine
#SBATCH --time=3-00:00            # Runtime in D-HH:MM
##SBATCH --partition=bethge
##SBATCH --partition=gpu-2080ti    # Partition to submit to
#SBATCH --partition=gpu-2080ti-preemptable    # Partition to submit to
#SBATCH --output=slurm/%x_%j.out  # File to which STDOUT will be written
#SBATCH --error=slurm/%x_%j.err   # File to which STDERR will be written
#SBATCH --gres=gpu:1              # Request one GPU
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=10000

# include information about the job in the output
scontrol show job=$SLURM_JOB_ID

log=/mnt/qb/work/bethge/ysharma/projects/objects_identifiability/logs/final_$1_$2

job_id=$2;

#--permutation --sobs-dim 20 --hidden-dim 120 --resample

if [ "$job_id" == 0 ]; then
    echo "ind. 22";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 2 --slot-dim 2 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 1 ]; then
    echo "ind. 33";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 3 --slot-dim 3 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 2 ]; then
    echo "ind. 55";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 5 --slot-dim 5 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 3 ]; then
    echo "dep. 22";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 2 --slot-dim 2 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 4 ]; then
    echo "dep. 33";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 3 --slot-dim 3 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 5 ]; then
    echo "dep. 55";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 5 --slot-dim 5 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 6 ]; then
    echo "ind. 22";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 2 --slot-dim 2 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --resample 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 7 ]; then
    echo "ind. 33";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 3 --slot-dim 3 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --resample 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 8 ]; then
    echo "ind. 55";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 5 --slot-dim 5 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --resample 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 9 ]; then
    echo "dep. 22";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 2 --slot-dim 2 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence --resample 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 10 ]; then
    echo "dep. 33";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 3 --slot-dim 3 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence --resample 2> ${log}.err | tee ${log}.log"
elif [ "$job_id" == 11 ]; then
    echo "dep. 55";
    srun singularity exec --nv --bind $WORK --writable-tmpfs /mnt/qb/work/bethge/ysharma/sifs/ssl_video_amazon.sif /bin/bash -c "python3 -u /mnt/qb/work/bethge/ysharma/projects/objects_identifiability/simulations.py --cuda --save-model --seed=$1 --n-slots 5 --slot-dim 5 --permutation --sobs-dim 20 --hidden-dim 120 --min-rec --statistical-dependence --resample 2> ${log}.err | tee ${log}.log"
fi
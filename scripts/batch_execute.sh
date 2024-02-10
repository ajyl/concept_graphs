#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..9}
do
#for k in 0 1 2 
#for k in 3 4 5 6 7
#for k in 0 2 3 4 5 6 7
#for k in 11 10 9 8 7 6 5 4 3 2 1 0
#for k in 9
for k in 8 7 6 5 4 3 2 1 0
do
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/fig2a_sbatch.sh $i $k
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/fig2b_sbatch.sh $i $k
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/perm_sbatch.sh $i $k
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/sprite_sbatch.sh $i $k
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/fix_sbatch.sh $i $k
      #sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/fix2_sbatch.sh $i $k
      sbatch /mnt/qb/work/bethge/ysharma/jack_scripts/final_sbatch.sh $i $k

done
done
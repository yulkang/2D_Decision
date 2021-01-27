#!/bin/sh
#
# Simple Matlab submit script for Slurm.
#
#
#SBATCH -A zidw                # The account name for the job.
#SBATCH -J ATT_DTB_VBD           # The job name.
#SBATCH -t 720:00                  # The time the job will take to run.
#SBATCH --mem-per-cpu=24gb        # The memory the job will use per cpu core.
#SBATCH -c 18 # number of cores

module load matlab/2019b

echo "Launching an Matlab run"
date


#Command to execute Matlab code
matlab -nosplash -nodisplay -nodesktop -r "run_fit_2D()" # > matoutfile

# End of script

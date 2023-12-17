#!/bin/bash

#SBATCH -A shakeri-lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH -c 1
#SBATCH -t 72:00:00
#SBATCH -o ./logs/train_%A.out
#SBATCH -e ./logs/train_%A.err

# link log files to respective directories
# requires that the runner script has certain flags in a certain order:
# order = --env env --alg alg --custom_reward reward
old_log=./logs/train_${SLURM_JOB_ID}.out
old_err=./logs/train_${SLURM_JOB_ID}.err
new_log=./evals/$5_$3_$7/train.out
new_err=./evals/$5_$3_$7/train.err

# set up environment
CONDA_ENV_PATH="/scratch/jgh2xh/envs/rlzoo"
PIP_EX="$CONDA_ENV_PATH/bin/pip"
PYTHON_EX="$CONDA_ENV_PATH/bin/python"

echo "Linking $old_log --> $new_log"
echo "Linking $old_err --> $new_err"
echo

ln -f $old_log $new_log
ln -f $old_err $new_err

echo $PYTHON_EX $@
$PYTHON_EX $@

echo
echo "Removing $old_log"
echo "Removing $old_err"
echo
rm $old_log
rm $old_err

echo "All done"
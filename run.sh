#!/bin/sh
#SBATCH --time=168:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --job-name=<job_name>
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=<your_email>
#SBATCH --output=/work/netthinker/shared/out_files/<your_name>/<file>.out

export PYTHONPATH=$WORK/tf-gpu-pkgs
module load singularity
singularity exec docker://lordvoldemort28/pytorch-opencv:dev python -u $@
#!/bin/bash
#SBATCH --job-name=icarl_5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24GB
#SBATCH --time=12:00:00

# the required parameter is just the prefix for storing the model
if [ $# -ne 1 ]; then
  exit
fi


# clean the module environment that we may have inherited from the calling session
ml purge

# load the relevant modules
ml PyTorch/0.4.0-gomkl-2018b-Python-2.7.15-CUDA-9.2.88
ml torchvision/0.2.1-gomkl-2018b-Python-2.7.15-CUDA-9.2.88 

echo 'Starting job'
# run the script
python main.py "$1"


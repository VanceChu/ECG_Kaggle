#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --time=01:30:00
#SBATCH --partition=short
#SBATCH --job-name=ecg_gen

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/envs/ecg_gen
conda create --prefix $CONPREFIX python=3.10

# Activate your environment
source activate $CONPREFIX

# Install packages...
pip install -r requirements.txt
pip install tensorflow[and-cuda]
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
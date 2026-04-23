#!/bin/bash -l
#SBATCH -t 10:00:00
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --output=./out/%x_%j_%a.out
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bert.depoorter@student.kuleuven.be
#SBATCH -A lp_lisagw
#SBATCH --job-name=test_1

module load GCC/11.3.0 GSL CUDA/12 FFTW/3.3.10-GCC-11.3.0 
conda activate lisatools_env
nvidia-smi

HOME_FOLDER=/data/leuven/367/vsc36785/LISA/few-mojito-review/PE_tests
SAMPLE_SCRIPTS=$HOME_FOLDER/PE_test_cases_response.py
INFERENCE_PARAMS=$HOME_FOLDER/config/config_test_1.yaml

python $SAMPLE_SCRIPTS --inference_params=$INFERENCE_PARAMS --cluster=vsc

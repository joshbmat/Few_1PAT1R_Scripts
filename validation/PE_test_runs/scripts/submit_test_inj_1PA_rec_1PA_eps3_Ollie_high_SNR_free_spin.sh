#!/bin/bash -l
#SBATCH -t 20:00:00
#SBATCH --cluster=mindwell
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --partition=gpu_b200
#SBATCH --gpus-per-node=1
#SBATCH --output=./out/%x_%j_%a.out
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=bert.depoorter@student.kuleuven.be
#SBATCH -A lp_mindwell_pilot
#SBATCH --job-name=test_inj_1PA_rec_1PA_eps3_Ollie_high_SNR_free_spin

module load GCC/11.3.0 GSL CUDA/12 FFTW/3.3.10-GCC-11.3.0
conda activate emri_env_ddpc
nvidia-smi

HOME_FOLDER=/data/leuven/367/vsc36785/LISA/Few_1PAT1R_Scripts/validation/PE_test_runs
SAMPLE_SCRIPTS=$HOME_FOLDER/PE_response.py
INFERENCE_PARAMS=$HOME_FOLDER/config/test_inj_1PA_rec_1PA_eps3_Ollie_high_SNR_free_spin.yaml

python $SAMPLE_SCRIPTS --config=$INFERENCE_PARAMS

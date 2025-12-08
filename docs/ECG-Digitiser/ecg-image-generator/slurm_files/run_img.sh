#! /bin/bash 
#SBATCH --nodes=1
#SBATCH --mem=250G
#SBATCH --ntasks-per-node=28
#SBATCH --time=36:10:00
#SBATCH --partition=long
#SBATCH --job-name=6_nq_nh

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wolf6245@ox.ac.uk

module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/ecg_gen
conda info --env

python gen_ecg_images_from_data_batch.py \
    -i /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared/06000 \
    -o /data/inet-multimodal-ai/wolf6245/data/ptb-xl/records500_prepared_w_images/06000 \
    -se 10 \
    --mask_unplotted_samples \
    --print_header \
    --store_config 2 \
    --lead_name_bbox \
    --lead_bbox \
    --random_print_header 0.7 \
    --calibration_pulse 0.6 \
    --wrinkles \
    --augment \
    -rot 5 \
    --num_images_per_ecg 4 \
    --random_bw 0.1

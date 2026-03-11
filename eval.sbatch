#!/bin/bash
#SBATCH --job-name=FREED_Eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --time=3-00:00           
#SBATCH --output=logs/freed-%j.out
#SBATCH --error=logs/freed-%j.err

source ~/.bashrc
conda activate freed

PROJ_DIR="/home/semakin_grisha/FFREED"
SOURCE_DIR="$PROJ_DIR/ffreed"
DATA_DIR="$SOURCE_DIR/data"
EXP_DIR="$PROJ_DIR/experiments"

BOX_CENTER="10.131, 41.879, 32.097"
BOX_SIZE="50.673, 50.198, 50.362"

python -u $SOURCE_DIR/main.py \
    --exp_root $EXP_DIR \
    --alert_collections $DATA_DIR/alert_collection.csv \
    --fragments $DATA_DIR/motifs/merged_motifs.json \
    --receptor $DATA_DIR/receptors/protein.pdbqt \
    --vina_program $PROJ_DIR/utils/qvina02 \
    --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1" \
    --fragmentation brics \
    --num_sub_proc 12 \
    --n_conf 3 \
    --exhaustiveness 8 \
    --save_freq 50 \
    --epochs 200 \
    --commands "evaluate" \
    --reward_version soft \
    --box_center "$BOX_CENTER" \
    --box_size "$BOX_SIZE" \
    --seed 150 \
    --timeout_dock 90 \
    --timeout_gen3d 30 \
    --name ffreed_expert_v1


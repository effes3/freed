#!/bin/bash
#SBATCH --job-name=FREED_Synthesis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=3-00:00           
#SBATCH --output=logs/freed-%j.out
#SBATCH --error=logs/freed-%j.err

# 1. Активация твоей среды (убедись, что имя совпадает)
source ~/.bashrc
conda activate freed

# 2. Переменные путей (замени /home/user на свой путь в системе)
PROJ_DIR="/home/semakin_grisha/FFREED"
SOURCE_DIR="$PROJ_DIR/ffreed"
DATA_DIR="$SOURCE_DIR/data"
EXP_DIR="$PROJ_DIR/experiments"

# 3. ПАРАМЕТРЫ ДОКИНГА (ВАЖНО!)
# Замени x,y,z на реальные координаты центра кармана белка
# Если ты их не знаешь, докинг будет работать впустую
BOX_CENTER="17.179,35.807,33.795" \
BOX_SIZE="20.0,20.0,20.0" \

# 4. ЗАПУСК
# --fragments: указываем твой новый JSON
# --starting_smile: меняем метки на те, что в твоем JSON (например, 15 для азида, 18 для галогенида)
python $SOURCE_DIR/main.py \
    --exp_root $EXP_DIR \
    --alert_collections $DATA_DIR/alert_collection.csv \
    --fragments $DATA_DIR/motifs/fragments_fixed.json \
    --receptor $DATA_DIR/receptors/protein.pdbqt \
    --vina_program $PROJ_DIR/utils/qvina02 \
    --starting_smile "c1([15*])ccc([18*])cc1" \
    --fragmentation brics \
    --num_sub_proc 12 \
    --n_conf 1 \
    --exhaustiveness 1 \
    --save_freq 50 \
    --epochs 200 \
    --commands "train,sample" \
    --reward_version soft \
    --box_center "$BOX_CENTER" \
    --box_size "$BOX_SIZE" \
    --seed 150 \
    --name freed_expert_v1  

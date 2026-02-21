#!/usr/bin/env bash
#SBATCH --job-name=pde4_lamsweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:1,lscratch:100
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=120:00:00
#SBATCH --array=0-3


SRC_PATH="./src"

# Original data path
DATA_PATH="data_for_training.parquet"

# Cache split dataset and save to the path
CACHED_PATH="cached_data_path" 

# Path to save checkpoints pt.
SAVE_DIR="save_path" 
mkdir -p "${SAVE_DIR}"

TRAINER="train.trainer"  # Trainer path
CHECKPOINT="./checkpoint/checkpoint_best_pcqm4mv2.pt" # pretrained graphormer
# CHECKPOINT="./checkpoints/best.pt"  # if you need to continue training your own model using the best checkpoint

# =========================================================
#   Run Different Seeds And Different Lambdas In Parallel
# =========================================================
# LAMBDAS=(0.12 0.16 0.20 0.24 0.28)  # for different LAMBDA's training. Lambda is the weight of selectivity loss, ranging (0, 1)
LAMBDAS=0.20
#SEEDS=123
SEEDS=(456 500 789 1000)  # Training with different seeds to estimate uncertainty

# make sure  SBATCH --array=0-(S*L-1). This allows parallel training
S=${#SEEDS[@]}   # Number of seeds
L=${#LAMBDAS[@]}  # Number of lambdas

# Custom unique training name
tid=${SLURM_ARRAY_TASK_ID}
seed_idx=$(( tid % S ))
lambda_idx=$(( tid / S ))

SEED=${SEEDS[$seed_idx]}
LAMBDA_AUX=${LAMBDAS[$lambda_idx]}
LAM_TAG="${LAMBDA_AUX//./p}"
SEED_TAG="seed${SEED}"


# ========================================================
#      Assign Run Name Based on Lambda and Seed
# ========================================================
RUN_NAME="graphormer_pde4_regress_lam${LAM_TAG}_${SEED_TAG}"

echo "SLURM_ARRAY_TASK_ID=${tid}"
echo "lambda_idx=${lambda_idx} lambda_aux=${LAMBDA_AUX}"
echo "seed_idx=${seed_idx} seed=${SEED}"
echo "run_name=${RUN_NAME}"

# Print Device
nvidia-smi || true

# ========================================================
#       Environment and Directory Setup: Important !!
# ========================================================
source "${ROOT_PATH}/GraphormerActSel/bin/activate"   # You must revise here to activate your environment accordingly
export PYTHONPATH="${SRC_PATH}:${PYTHONPATH:-}"
cd "${SRC_PATH}"

CACHED_DATASET_PATH="${CACHED_PATH}/processed/data.pt"


ARGS=(
  -u -m "${TRAINER}"
  --data_path "${DATA_PATH}"
  --root_path "${CACHED_PATH}"
  --save_dir "${SAVE_DIR}"
  --run_name "${RUN_NAME}"
  --checkpoint "${CHECKPOINT}"
  --seed "${SEED}"
  --epochs 100
  --batch_size 16
  --num_workers 4
  --grad_clip 1.0
  --lambda_aux "${LAMBDA_AUX}"
  --huber_delta 1.0
  --patience 20
  --min_delta 0.0
  --scheduler cosine
  --tmax 50
  --encoder_lr 2e-5
  --adaptor_lr 1e-4
  --weight_decay 0.01
  --cached_dataset_path "${CACHED_DATASET_PATH}"
)

echo "Running: python ${ARGS[*]}"
python "${ARGS[@]}"

#!/usr/bin/env bash

# Configurable parameters (override via environment variables)
MODEL_SIZE=${MODEL_SIZE:-small}               # tiny|base|small|medium|large-v3 etc.
MODEL_NAME=${MODEL_NAME:-openai/whisper-$MODEL_SIZE}
LANGUAGE=${LANGUAGE:-vietnamese}
SAMPLING_RATE=${SAMPLING_RATE:-16000}
NUM_PROC=${NUM_PROC:-4}
TRAIN_STRATEGY=${TRAIN_STRATEGY:-steps}
LEARNING_RATE=${LEARNING_RATE:-5e-6}
WARMUP=${WARMUP:-100}
TRAIN_BATCHSIZE=${TRAIN_BATCHSIZE:-48}
EVAL_BATCHSIZE=${EVAL_BATCHSIZE:-32}
NUM_STEPS=${NUM_STEPS:-1000}
NUM_EPOCHS=${NUM_EPOCHS:-20}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-output_models}
TRAIN_DATASETS=${TRAIN_DATASETS:-./data/user_train.parquet}
EVAL_DATASETS=${EVAL_DATASETS:-./data/user_eval.parquet}

mkdir -p "$OUTPUT_BASE_DIR"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/whisper"

if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "Output directory $OUTPUT_DIR already exists and is non-empty. Skipping training."
    exit 0
fi
    
echo "Starting training..."
echo "Output directory: $OUTPUT_DIR"

python train_asr.py \
    --model_name "$MODEL_NAME" \
    --language "$LANGUAGE" \
    --sampling_rate $SAMPLING_RATE \
    --num_proc $NUM_PROC \
    --train_strategy "$TRAIN_STRATEGY" \
    --learning_rate $LEARNING_RATE \
    --warmup $WARMUP \
    --train_batchsize $TRAIN_BATCHSIZE \
    --eval_batchsize $EVAL_BATCHSIZE \
    --num_steps $NUM_STEPS \
    --num_epochs $NUM_EPOCHS \
    --output_dir "$OUTPUT_DIR" \
    --train_datasets "$TRAIN_DATASETS" \
    --eval_datasets "$EVAL_DATASETS"
    
echo "Finished training!"
echo "-------------------------------------------------------"

echo "All training completed!"
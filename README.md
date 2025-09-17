## MMedFD:A Real-world Healthcare Benchmark for Multi-turn Full-Duplex Automatic Speech Recognition

## Abstract
Automatic speech recognition (ASR) in clinical dialogue demands robustness to full-duplex interaction, speaker overlap,and low-latency constraints, yet open benchmarks remain scarce. We present MMedFD, the first real-world Chinese healthcare ASR corpus designed for multi-turn, full-duplex settings. Captured from a deployed AI assistant, the dataset comprises 5,805 annotated sessions with synchronized user and mixed-channel views, RTTM/CTM timing, and role la-
bels. We introduce a model-agnostic pipeline for streaming segmentation, speaker attribution, and dialogue memory, and
fine-tune Whisper-small on role-concatenated audio for long context recognition. ASR evaluation includes WER, CER,
and HC-WER, which measures concept-level accuracy across
healthcare settings. LLM-generated responses are assessed
using rubric-based and pairwise protocols. MMedFD estab-
lishes a reproducible framework for benchmarking streaming
ASR and end-to-end duplex agents in healthcare deployment.
The dataset and related resources are publicly available at

## Whisper ASR Fine-tuning and Inference

This repository provides a streamlined workflow to fine-tune OpenAI Whisper models on your data, run inference on a dataset, and evaluate results with CER/WER. The codebase is written in English and exposes consistent CLI options. Whisper model size is configurable.

### Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU is supported but slow)
- Install dependencies:

```bash
pip install -U transformers datasets evaluate soundfile pandas pyarrow torch accelerate
```

### Data Format

- Training/evaluation data is expected as Parquet files with at least two columns:
  - `audio`: a struct containing raw WAV bytes under key `bytes`
  - `text`: the reference transcription
- Example paths used below:
  - Train: `./data/user_train.parquet`
  - Eval/Test: `./data/user_eval.parquet`

### Train

Use the provided script to train. You can select Whisper size via `MODEL_SIZE` or set `MODEL_NAME` directly.

```bash
# Example: small model, Vietnamese language tokens
export MODEL_SIZE=small
export LANGUAGE=vietnamese
bash run_train_asr.sh
```

Key environment variables (override as needed):
- `MODEL_SIZE`: tiny|base|small|medium|large-v3 (default: small)
- `MODEL_NAME`: full HF model id (default: openai/whisper-$MODEL_SIZE)
- `LANGUAGE`: language for processor tokens, e.g. chinese|vietnamese (default: vietnamese)
- `TRAIN_DATASETS`, `EVAL_DATASETS`: Parquet file paths
- `OUTPUT_BASE_DIR`: where checkpoints are saved (default: output_models)

Under the hood the script runs:

```bash
python train_asr.py \
  --model_name openai/whisper-small \
  --language vietnamese \
  --train_datasets ./data/user_train.parquet \
  --eval_datasets ./data/user_eval.parquet \
  --output_dir output_models/whisper \
  --train_strategy steps --num_steps 1000 --train_batchsize 48 --eval_batchsize 32 \
  --learning_rate 5e-6 --warmup 100 --num_proc 4
```

Notes:
- Processed features are cached in `processed_dataset_cache/` to speed up subsequent runs.
- Gradient checkpointing is enabled via training args for memory efficiency.

### Inference

Run ASR on a Parquet dataset or a fallback HF dataset. Outputs a CSV at `whisper_inference_results/predictions.csv` by default.

```bash
python whisper_asr_infer.py \
  --base_model_name openai/whisper-small \
  --checkpoint_dir output_models/whisper \
  --test_dataset_path ./data/user_eval.parquet \
  --output_dir whisper_inference_results \
  --chunk_length_s 30 --batch_size 8 \
  --language vietnamese
```

Arguments:
- `--base_model_name`: base Whisper model to load processor/tokenizer from
- `--checkpoint_dir`: path to your fine-tuned model directory; auto-picks latest `checkpoint-*`
- `--test_dataset_path`: Parquet with `audio.bytes` and (optionally) `text`. If missing, uses a Hugging Face dataset fallback
- `--output_dir`: where `predictions.csv` is written
- `--language`: language for processor tokens; optional

Output CSV columns:
- `ID`, `Original Text`, `Prediction` (+ optional language columns if present in source dataset)

### Evaluation

Compute CER and WER for your predictions.

```bash
python compute_score.py \
  --predict_path whisper_inference_results/predictions.csv \
  --groundtruth_path ./data/user_eval.parquet \
  --merge_on ID
```

This writes a summary to `whisper_inference_results/evaluation_results.txt` and prints CER/WER to stdout.

### Switching Model Sizes

Pick a size by setting `MODEL_SIZE` in the train script or pass a full `--model_name`/`--base_model_name`:
- Sizes: `tiny`, `base`, `small`, `medium`, `large-v3`
- Examples:
  - Train with base: `MODEL_SIZE=base bash run_train_asr.sh`
  - Infer with medium: `--base_model_name openai/whisper-medium`

### Tips

- If your Parquet audio has multiple channels, the inference code converts to mono automatically.
- Ensure sample rate is 16 kHz (resample upstream if needed).
- For long audio files, adjust `--chunk_length_s` and `--batch_size` to balance speed and memory.

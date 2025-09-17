import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, load_dataset, concatenate_datasets
import re
import string
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig

parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--language', 
    type=str, 
    required=False, 
    default='vietnamese', 
    help="Language for processor special tokens (e.g., 'chinese', 'vietnamese')."
)
parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=2, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=48, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=32, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=20, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)
parser.add_argument(
    '--train_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default="./train_from_csv.parquet", 
    help='List of datasets to be used for training.'
)
parser.add_argument(
    '--eval_datasets', 
    type=str, 
    nargs='+', 
    required=True, 
    default="./eval_from_csv.parquet", 
    help='List of datasets to be used for evaluation.'
)

args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps or epoch.')


gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_lower_case = False
do_remove_punctuation = False
# punctuation_to_remove = string.punctuation.replace("'", "")
punctuation_to_remove_regex = f"[{re.escape(string.punctuation)}]"

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

# Note: enable gradient checkpointing via Seq2SeqTrainingArguments only to avoid
# "backward through the graph twice" errors.
if gradient_checkpointing:
    # Disable cache to be compatible with gradient checkpointing
    model.config.use_cache = False

# model.config.apply_spec_augment = True
# model.config.mask_time_prob = 0.05
# model.config.mask_feature_prob = 0.05

model.config.suppress_tokens = None

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = None

if gradient_checkpointing:
    model.config.use_cache = False

def prepare_dataset(batch):
    audio = batch["audio"]
    # Skip if audio is None or invalid
    if audio is None or not isinstance(audio, dict) or "array" not in audio or audio["array"] is None:
        # Set invalid values that will be filtered out
        batch["input_length"] = 0
        batch["input_features"] = [0] * 80 * 3000  # dummy features
        batch["labels"] = []
        batch["labels_length"] = 0
        return batch
    
    # Handle numpy arrays properly
    audio_array = audio["array"]
    if isinstance(audio_array, np.ndarray):
        audio_array = audio_array.astype(np.float32)
    
    batch["input_length"] = len(audio_array)
    batch["input_features"] = feature_extractor(audio_array, sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # optional pre-processing steps
    transcription = batch["text"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = re.sub(punctuation_to_remove_regex, " ", transcription).strip()
    batch["labels"] = tokenizer(transcription).input_ids
    batch["labels_length"] = len(tokenizer(transcription, add_special_tokens=False).input_ids)
    return batch

def filter_labels(labels_length):
        """Filter label sequences longer than max length (448)"""
        return labels_length < 448
    
print('DATASET PREPARATION IN PROGRESS...')
import pandas as pd
from datasets import Dataset
import io
import soundfile as sf
import numpy as np
import pyarrow.parquet as pq

# Read parquet files with nested data handling
def read_parquet_files(file_paths):
    # Handle single file or list of files
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    all_dfs = []
    for file_path in file_paths:
        # Use ParquetFile to read in batches to avoid nested data issues
        parquet_file = pq.ParquetFile(file_path)
        
        # Read in smaller batches and combine
        batches = []
        for batch in parquet_file.iter_batches(batch_size=100, use_pandas_metadata=True):
            df_batch = batch.to_pandas()
            batches.append(df_batch)
        
        # Combine all batches for this file
        df = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        all_dfs.append(df)
    
    # Combine all files
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

train_df = read_parquet_files(args.train_datasets)
eval_df = read_parquet_files(args.eval_datasets)

# Convert audio bytes to proper format
def convert_audio_bytes(df):
    audio_data = []
    for _, row in df.iterrows():
        if row['audio'] is not None and isinstance(row['audio'], dict) and 'bytes' in row['audio']:
            try:
                # Read WAV bytes
                audio_bytes = row['audio']['bytes']
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                audio_data.append({
                    'array': audio_array,
                    'sampling_rate': sample_rate
                })
            except Exception as e:
                print(f"Error processing audio: {e}")
                audio_data.append(None)
        else:
            audio_data.append(None)
    df['audio'] = audio_data
    return df

train_df = convert_audio_bytes(train_df)
eval_df = convert_audio_bytes(eval_df)

# Filter out None audio entries
train_df = train_df[train_df['audio'].notna()]
eval_df = eval_df[eval_df['audio'].notna()]

raw_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "eval": Dataset.from_pandas(eval_df)
})
# Check if processed dataset exists
import os
processed_dataset_path = "processed_dataset_cache"

if os.path.exists(processed_dataset_path):
    print(f"Loading processed dataset from cache: {processed_dataset_path}")
    raw_dataset = DatasetDict.load_from_disk(processed_dataset_path)
else:
    print("Processing dataset...")
    # No need to cast_column since audio is already in the right format
    raw_dataset = raw_dataset.map(prepare_dataset, remove_columns=raw_dataset.column_names["train"], num_proc=args.num_proc)
    
    # Filter out invalid audio entries (where input_length is 0)
    raw_dataset = raw_dataset.filter(lambda x: x > 0, num_proc=1, input_columns=['input_length'])
    raw_dataset = raw_dataset.filter(filter_labels, num_proc=1, input_columns=['labels_length'])
    
    # Save the processed dataset
    print(f"Saving processed dataset to cache: {processed_dataset_path}")
    raw_dataset.save_to_disk(processed_dataset_path)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")
def compute_metrics(eval_pred):
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids

    # Decode predictions and labels to text
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Filter out empty references and their corresponding predictions
    valid_pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip()]
    
    if not valid_pairs:
        print("Warning: All references are empty strings!")
        return {"wer": float("inf")}  # or another suitable default value
        
    filtered_preds, filtered_refs = zip(*valid_pairs)
    
    # Compute WER on valid pairs
    try:
        wer = 100 * metric.compute(predictions=filtered_preds, references=filtered_refs)
        return {"wer": wer}
    except Exception as e:
        print(f"Error computing WER: {e}")
        # Log some debugging information
        print(f"Number of predictions: {len(filtered_preds)}")
        print(f"Number of references: {len(filtered_refs)}")
        print("Sample predictions:", filtered_preds[:5])
        print("Sample references:", filtered_refs[:5])
        return {"wer": float("inf")}


# For older transformers versions, avoid unsupported args and keep core training config only
if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        fp16=True,
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        generation_max_length=448,
        logging_steps=500
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        fp16=True,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        generation_max_length=225,
        logging_steps=50
    )

# Create and save GenerationConfig
gen_config = GenerationConfig(
    max_length=448,
    begin_suppress_tokens=[220, 50257]
)
gen_config.save_pretrained(args.output_dir)

# For older transformers versions, Seq2SeqTrainer may not support generation_config.
# Assign generation config directly to the model.
model.generation_config = gen_config

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()
print('DONE TRAINING')

import os
import re
import string
import argparse
import torch
import pandas as pd
import io
import soundfile as sf
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer, WhisperFeatureExtractor, GenerationConfig, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def get_args():
    parser = argparse.ArgumentParser(description="Whisper ASR inference with configurable model size and inputs.")
    parser.add_argument("--base_model_name", type=str, default="openai/whisper-small", help="Base Whisper model to load processor/tokenizer from (e.g., openai/whisper-small)")
    parser.add_argument("--checkpoint_dir", type=str, default="output_models/whisper", help="Directory containing fine-tuned checkpoints")
    parser.add_argument("--test_dataset_path", type=str, default="./data/user_eval.parquet", help="Path to test dataset parquet file. If missing, a fallback HF dataset is used")
    parser.add_argument("--output_dir", type=str, default="whisper_inference_results", help="Directory to write predictions CSV into")
    parser.add_argument("--chunk_length_s", type=int, default=30, help="Chunk length in seconds for long audio in pipeline")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference in pipeline")
    parser.add_argument("--language", type=str, default=None, help="Language for processor special tokens (e.g., 'chinese', 'vietnamese'). If None, do not set")
    return parser.parse_args()

def resolve_model_path(checkpoint_dir: str) -> str:
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*'))
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        return latest_checkpoint
    return checkpoint_dir

def load_processor_and_tokenizer(base_model_name: str, language: str | None):
    # Prefer loading from base model to avoid missing config issues
    if language:
        processor = WhisperProcessor.from_pretrained(base_model_name, language=language, task="transcribe")
    else:
        processor = WhisperProcessor.from_pretrained(base_model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_name)
    tokenizer = WhisperTokenizer.from_pretrained(base_model_name)
    return processor, feature_extractor, tokenizer

def build_pipeline(model_path: str, processor: WhisperProcessor, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer, chunk_length_s: int, batch_size: int):
    print(f"Loading trained model weights from: {model_path}")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)

    # Model generation settings
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = None

    gen_config = GenerationConfig(
        max_length=448,
        begin_suppress_tokens=[220, 50257],
        no_timestamps_token_id=tokenizer.convert_tokens_to_ids("<|notimestamps|>") if "<|notimestamps|>" in tokenizer.get_vocab() else 50363,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.convert_tokens_to_ids("<|startoftranscript|>") if "<|startoftranscript|>" in tokenizer.get_vocab() else 50258
    )
    model.generation_config = gen_config

    print("Building ASR pipeline...")
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size
    )

# Read parquet files with nested data handling
def read_parquet_files(file_paths):
    """Read parquet files with nested data handling"""
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    all_dfs = []
    for file_path in file_paths:
        parquet_file = pq.ParquetFile(file_path)
        batches = []
        for batch in parquet_file.iter_batches(batch_size=100, use_pandas_metadata=True):
            df_batch = batch.to_pandas()
            batches.append(df_batch)
        df = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
        all_dfs.append(df)
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def convert_audio_bytes(df):
    """Convert audio bytes to proper format"""
    audio_data = []
    for idx, row in df.iterrows():
        if row['audio'] is not None and isinstance(row['audio'], dict) and 'bytes' in row['audio']:
            try:
                audio_bytes = row['audio']['bytes']
                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                # Ensure audio is 1D mono
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                # Force float32 dtype
                audio_array = audio_array.astype(np.float32)
                audio_data.append({
                    'array': audio_array,
                    'sampling_rate': sample_rate
                })
            except Exception as e:
                print(f"Error processing audio at index {idx}: {e}")
                audio_data.append(None)
        else:
            audio_data.append(None)
    df['audio'] = audio_data
    return df

def load_dataset_for_inference(test_dataset_path: str):
    if os.path.exists(test_dataset_path):
        test_df = read_parquet_files(test_dataset_path)
        test_df = convert_audio_bytes(test_df)
        test_df = test_df[test_df['audio'].notna()]
        return Dataset.from_pandas(test_df)
    print(f"Warning: Test dataset {test_dataset_path} not found. Using default dataset.")
    ds = load_dataset('wnkh/MultiMed', 'Vietnamese', split='corrected.test')
    return ds.cast_column("audio", Audio(sampling_rate=16000))

def normalize_text_to_chars(text):
    """Removes punctuation and extra spaces from text."""
    if isinstance(text, str):
        return " ".join(re.sub(f"[{re.escape(string.punctuation)}]", " ", text).split())
    return text

def transcribe_audio(batch):
    """Transcribes audio using the Whisper model."""
    try:
        if isinstance(batch["audio"], dict) and "array" in batch["audio"]:
            # Parquet-style audio dict with raw bytes converted to array
            audio_input = batch["audio"]["array"]
            sampling_rate = batch["audio"].get("sampling_rate", 16000)
            
            # Validate audio
            if audio_input is None:
                print("Warning: audio_input is None")
                batch["prediction"] = ""
                return batch
            
            # Ensure numpy array
            if not isinstance(audio_input, np.ndarray):
                if isinstance(audio_input, (list, tuple)):
                    audio_input = np.array(audio_input, dtype=np.float32)
                else:
                    print(f"Warning: Unexpected audio type: {type(audio_input)}")
                    batch["prediction"] = ""
                    return batch
            
            # Force float32 dtype
            if audio_input.dtype != np.float32:
                audio_input = audio_input.astype(np.float32)
            
            # Ensure 1D mono
            if len(audio_input.shape) > 1:
                audio_input = audio_input.mean(axis=1)
            
            # Validate non-empty audio
            if len(audio_input) == 0:
                print("Warning: Empty audio array")
                batch["prediction"] = ""
                return batch
                
        else:
            # Hugging Face dataset style
            audio_input = batch["audio"]
            
            # If dict with path/array
            if isinstance(audio_input, dict) and "path" in audio_input:
                audio_input = audio_input.get("array", audio_input)
        
        # Run ASR
        result = pipe(audio_input)
        batch["prediction"] = result['text'] if result else ""
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        print(f"Audio type: {type(batch.get('audio'))}")
        if isinstance(batch.get('audio'), dict):
            print(f"Audio keys: {batch['audio'].keys()}")
        batch["prediction"] = ""
    
    return batch

def infer_and_save_to_csv(dataset):
    """Processes the dataset, transcribes audio, and saves predictions to a CSV file."""
    predictions = []
    
    # Iterate one-by-one to avoid memory pressure
    for batch in tqdm(dataset, desc="Transcribing"):
        result = transcribe_audio(batch)
        predictions.append(result)
    
    # Build output DataFrame
    df_data = {
        "ID": [f"sentence_{i+1}" for i in range(len(predictions))],
        "Original Text": [batch.get("text", "") for batch in predictions],
        "Prediction": [batch.get("prediction", "") for batch in predictions]
    }
    
    # Add optional multilingual text fields when present
    for field in ["French", "Chinese", "German", "Vietnamese"]:
        if any(field in batch for batch in predictions):
            df_data[field] = [batch.get(field, "") for batch in predictions]
    
    df = pd.DataFrame(df_data)
    return df

def main():
    args = get_args()

    print(f"Using device: {device} | dtype: {torch_dtype}")
    model_path = resolve_model_path(args.checkpoint_dir)
    print(f"Loading model from: {model_path}")
    print(f"Loading processor from base model: {args.base_model_name}")
    processor, feature_extractor, tokenizer = load_processor_and_tokenizer(args.base_model_name, args.language)

    global pipe
    pipe = build_pipeline(model_path, processor, feature_extractor, tokenizer, args.chunk_length_s, args.batch_size)

    dataset = load_dataset_for_inference(args.test_dataset_path)
    df = infer_and_save_to_csv(dataset)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Predictions and corresponding texts saved to {output_file}")

if __name__ == "__main__":
    main()
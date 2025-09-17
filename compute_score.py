import pandas as pd
import re
import string
import argparse
from evaluate import load

def normalize_text(text):
    """Normalize text by removing punctuation and spacing out characters."""
    punctuation_to_remove_regex = f"[{re.escape(string.punctuation)}]"
    text = re.sub(punctuation_to_remove_regex, " ", text).strip() if isinstance(text, str) else text
    return " ".join(text) if isinstance(text, str) else text

def compute_asr_scores(reference_texts, predicted_texts):
    """Compute CER and WER between references and predictions."""
    cer_metric = load("cer")
    wer_metric = load("wer")

    references = [normalize_text(text) for text in reference_texts]
    predictions = [normalize_text(text) for text in predicted_texts]

    cer_score = cer_metric.compute(references=references, predictions=predictions)
    wer_score = wer_metric.compute(references=references, predictions=predictions)

    return cer_score, wer_score

# === Parse Arguments ===
parser = argparse.ArgumentParser(description='Compute ASR scores (CER and WER) for predictions.')
parser.add_argument(
    '--predict_path', 
    type=str, 
    required=False,
    default='whisper_inference_results/predictions.csv',
    help='Path to predictions CSV file'
)
parser.add_argument(
    '--groundtruth_path', 
    type=str, 
    required=False,
    default='./test_from_csv.parquet',
    help='Path to ground truth file (CSV or Parquet)'
)
parser.add_argument(
    '--merge_on', 
    type=str, 
    required=False,
    default='ID',
    help='Column name to merge predictions and ground truth on'
)

args = parser.parse_args()

# === Load Predictions and Ground Truth ===
predict_path = args.predict_path
groundtruth_path = args.groundtruth_path

# Load predictions
predict_df = pd.read_csv(predict_path)

# Load ground-truth labels (supports CSV and Parquet)
if groundtruth_path.endswith('.parquet'):
    import pyarrow.parquet as pq
    # Use the same batched parquet reading logic as inference
    parquet_file = pq.ParquetFile(groundtruth_path)
    batches = []
    for batch in parquet_file.iter_batches(batch_size=100, use_pandas_metadata=True):
        df_batch = batch.to_pandas()
        batches.append(df_batch)
    label_df = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()
    # Add ID column for merging when missing
    if 'ID' not in label_df.columns:
        label_df['ID'] = [f"sentence_{i+1}" for i in range(len(label_df))]
else:
    label_df = pd.read_csv(groundtruth_path)

# Merge predictions with ground truth
merge_column = args.merge_on
if merge_column not in label_df.columns or merge_column not in predict_df.columns:
    # Fallback to index-based merge
    print(f"Warning: Merge column '{merge_column}' not found. Merging by index.")
    # Ensure equal length
    min_len = min(len(label_df), len(predict_df))
    label_df = label_df.iloc[:min_len].reset_index(drop=True)
    predict_df = predict_df.iloc[:min_len].reset_index(drop=True)
    merged_df = pd.concat([label_df, predict_df], axis=1)
else:
    merged_df = pd.merge(label_df, predict_df, on=merge_column, suffixes=('_label', '_predict'))

# Get reference and prediction text columns with flexible fallbacks
if 'text' in merged_df.columns:
    references = merged_df['text'].tolist()
elif 'Original Text' in merged_df.columns:
    references = merged_df['Original Text'].tolist()
elif 'text_label' in merged_df.columns:
    references = merged_df['text_label'].tolist()
else:
    raise ValueError("Cannot find reference text column. Available columns: " + str(merged_df.columns.tolist()))

if 'Prediction' in merged_df.columns:
    predictions = merged_df['Prediction'].tolist()
elif 'prediction' in merged_df.columns:
    predictions = merged_df['prediction'].tolist()
else:
    raise ValueError("Cannot find prediction column. Available columns: " + str(merged_df.columns.tolist()))

cer, wer = compute_asr_scores(references, predictions)

# === Print Results ===
print(f"\n=== ASR Evaluation Results ===")
print(f"Prediction file: {predict_path}")
print(f"Ground truth file: {groundtruth_path}")
print(f"Number of samples: {len(merged_df)}")
print(f"\nCharacter Error Rate (CER): {cer*100:.2f}%")
print(f"Word Error Rate (WER): {wer*100:.2f}%")

# 保存结果到文件
results = {
    'CER': f"{cer*100:.2f}%",
    'WER': f"{wer*100:.2f}%",
    'num_samples': len(merged_df)
}

import os
result_dir = os.path.dirname(predict_path)
result_file = os.path.join(result_dir, 'evaluation_results.txt')

with open(result_file, 'w') as f:
    f.write("=== ASR Evaluation Results ===\n")
    f.write(f"Prediction file: {predict_path}\n")
    f.write(f"Ground truth file: {groundtruth_path}\n")
    f.write(f"Number of samples: {len(merged_df)}\n")
    f.write(f"\nCharacter Error Rate (CER): {cer*100:.2f}%\n")
    f.write(f"Word Error Rate (WER): {wer*100:.2f}%\n")

print(f"\nResults saved to: {result_file}")

import torch
import numpy as np
import pandas as pd
from src.resnet_model import ResNet18
from preprocessing_pipeline.util import random_data_split
from sklearn.metrics import confusion_matrix
import os

# 1. Setup & Loading
MODEL_PATH = "./models/id_453_resnet_trial_221_32_1_dropout_0.5_adamw_lr_0.0005_wd_0.01_clip_length_3_increased_aug_append.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_LENGTH = 3

print(f"Loading model from {MODEL_PATH}...")
model = ResNet18().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["net_state_dict"])
model.eval()

# 2. Evaluation Loop
NUM_RUNS = 5
print(f"Running evaluation {NUM_RUNS} times to get stable results...")

all_run_metrics = []
all_imposter_stats = []
all_target_stats = []

for run in range(NUM_RUNS):
    print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")
    
    # 2.1 Data Loading
    print("Loading test data...")
    # random_data_split performs data split and returns tensors and labels
    data = random_data_split(path="./", clip_length=CLIP_LENGTH)
    test_x, test_y = data['test']
    test_labels = data['test_labels']

    # 2.2 Generate Predictions
    print("Evaluating model...")
    run_preds = []
    run_labels = []

    with torch.no_grad():
        # Batch processing to avoid memory issues if test set is large
        batch_size = 32
        for i in range(0, len(test_x), batch_size):
            batch_x = test_x[i:i+batch_size].to(DEVICE)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs > 0.5).int().cpu().numpy()
            run_preds.extend(preds)
            run_labels.extend(test_y[i:i+batch_size].numpy())

    run_preds = np.array(run_preds)
    run_labels = np.array(run_labels)

    # 2.3 Store Run Metrics
    cm = confusion_matrix(run_labels, run_preds)
    tn, fp, fn, tp = cm.ravel()
    
    global_far = fp / (fp + tn) if (fp + tn) > 0 else 0
    global_frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    all_run_metrics.append({
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'accuracy': accuracy,
        'global_far': global_far,
        'global_frr': global_frr
    })

    # 2.4 Imposter Analysis
    imposter_indices = np.where(run_labels == 0)[0]
    imposter_preds = run_preds[imposter_indices]
    imposter_speaker_ids = [test_labels[i] for i in imposter_indices]
    
    imposter_df = pd.DataFrame({
        'Speaker_ID': imposter_speaker_ids,
        'Is FA': (imposter_preds == 1).astype(int)
    })
    
    imposter_run_stats = imposter_df.groupby('Speaker_ID').agg(
        Total_Attempts=('Is FA', 'count'),
        False_Acceptances=('Is FA', 'sum')
    ).reset_index()
    all_imposter_stats.append(imposter_run_stats)

    # 2.5 Target Analysis
    target_indices = np.where(run_labels == 1)[0]
    target_preds = run_preds[target_indices]
    target_speaker_ids = [test_labels[i] for i in target_indices]
    
    target_df = pd.DataFrame({
        'Speaker_ID': target_speaker_ids,
        'Is FR': (target_preds == 0).astype(int)
    })
    
    target_run_stats = target_df.groupby('Speaker_ID').agg(
        Total_Attempts=('Is FR', 'count'),
        False_Rejections=('Is FR', 'sum')
    ).reset_index()
    all_target_stats.append(target_run_stats)

# 3. Aggregate Results
print("\n" + "="*30)
print("AGGREGATED RESULTS")
print("="*30)

# 3.1 Aggregate Global Metrics
metrics_df = pd.DataFrame(all_run_metrics)
mean_metrics = metrics_df.mean()

print("\n--- Analysis 1: Mean Confusion Matrix & Global Rates ---")
print(f"Mean TP: {mean_metrics['tp']:.1f}, TN: {mean_metrics['tn']:.1f}, FP: {mean_metrics['fp']:.1f}, FN: {mean_metrics['fn']:.1f}")
print(f"Mean Accuracy: {mean_metrics['accuracy']:.4f}")
print(f"Mean Global FAR: {mean_metrics['global_far']:.2%}")
print(f"Mean Global FRR: {mean_metrics['global_frr']:.2%}")

# 3.2 Aggregate Imposter Stats
combined_imposter = pd.concat(all_imposter_stats)
agg_imposter = combined_imposter.groupby('Speaker_ID').agg(
    Mean_Attempts=('Total_Attempts', 'mean'),
    Mean_False_Acceptances=('False_Acceptances', 'mean')
).reset_index()
agg_imposter['Mean Individual FAR'] = agg_imposter['Mean_False_Acceptances'] / agg_imposter['Mean_Attempts']

print("\n--- Analysis 2: Mean Top 5 Confusing Imposters ---")
top_5_imposters = agg_imposter.sort_values(by='Mean Individual FAR', ascending=False).head(5)
if top_5_imposters.empty:
    print("No Imposters found in the test set.")
else:
    print(top_5_imposters.to_string(index=False, formatters={
        'Mean_Attempts': '{:.1f}'.format,
        'Mean_False_Acceptances': '{:.1f}'.format,
        'Mean Individual FAR': '{:.2%}'.format
    }))

# 3.3 Aggregate Target Stats
combined_target = pd.concat(all_target_stats)
agg_target = combined_target.groupby('Speaker_ID').agg(
    Mean_Attempts=('Total_Attempts', 'mean'),
    Mean_False_Rejections=('False_Rejections', 'mean')
).reset_index()
agg_target['Mean Individual FRR'] = agg_target['Mean_False_Rejections'] / agg_target['Mean_Attempts']

print("\n--- Analysis 3: Mean Target Consistency (False Rejection Rates) ---")
if agg_target.empty:
    print("No Target samples found in the test set.")
else:
    print(agg_target.sort_values(by='Mean Individual FRR', ascending=False).to_string(index=False, formatters={
        'Mean_Attempts': '{:.1f}'.format,
        'Mean_False_Rejections': '{:.1f}'.format,
        'Mean Individual FRR': '{:.2%}'.format
    }))

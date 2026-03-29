# =========================================================
# EVA02 + XGBoost — Ocean Internal Wave Classification
# CPU VERSION
# =========================================================
# OUTPUT FILES (as written on server; renamed to pipeline_a_* in repo):
#   • python_cpu_evaluation_metrics.json  → pipeline_a_cpu_evaluation.json
#   • python_cpu_inference_metrics.json   → pipeline_a_cpu_inference.json
#
# FULL PIPELINE (no cached embeddings — raw image path every run):
#   1. Load raw PNG from disk
#   2. Resize 448×448, convert to RGB, normalise (timm transforms)
#   3. EVA-02 forward pass (timm model)
#   4. Extract 1000-dim softmax embedding
#   5. Run XGBoost classification
#   6. Output prediction
#
# BENCHMARKING PROTOCOL:
#   • Test set       : 5860 images
#   • Batch size     : 128
#   • Warm-up        : 1 FULL PASS through all 5860 test images (not measured)
#   • Profiling runs : 5 measured runs (mean ± std reported)
#   • Cooldown       : 30 s between consecutive runs
#   • Excluded       : model load, data preload, warm-up pass
# =========================================================

import os
import time
import json
import datetime
import psutil
import numpy as np
import pandas as pd
import torch
import timm
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier
############################################
# CONFIG
############################################

BASE_DIR      = "/tmp/benchmarks_iws"
DATA_DIR      = f"{BASE_DIR}/data/raw"
RESULTS_DIR   = f"{BASE_DIR}/cpu_python_iws/results"

TEST_IMG_DIR   = f"{DATA_DIR}/test"
TEST_CSV       = f"{DATA_DIR}/test.csv"

XGB_MODEL_PATH = "/tmp/benchmarks_iws/cpu_python_iws/results/xgboost_model.json"

BATCH_SIZE    = 128
NUM_WORKERS   = 4
DEVICE        = "cpu"
PIPELINE_NAME = "eva02_xgboost_iws"

COOLDOWN_S    = 30.0      # 30 s cooldown between consecutive runs
N_PROFILE_RUNS = 5        # 5 measured runs

os.makedirs(RESULTS_DIR, exist_ok=True)

############################################
# DATASET
# Steps 1 & 2: load raw PNG, resize 448x448,
# convert RGB, normalise via timm transforms
############################################

class ImageDataset(Dataset):
    def __init__(self, files, transform):
        self.files     = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")  # step 1 & 2
        return self.transform(img)                         # resize 448, normalise

############################################
# LOAD DATA
############################################

print(f"Loading data from: {DATA_DIR}")

test_df  = pd.read_csv(TEST_CSV)

# FIX 2: correct assertion — 5860 not 128
assert len(test_df) == 5860, f"Expected 5860 test images, got {len(test_df)}"

test_df["file"]  = test_df["id"].apply(lambda x: os.path.join(TEST_IMG_DIR,  x))

############################################
# LOAD EVA02 MODEL
# Step 3: timm model, NOT from .npy file
############################################

print("Loading EVA02 model...")
model = timm.create_model(
    "timm/eva02_large_patch14_448.mim_m38m_ft_in1k",
    pretrained=True
)
data_config = timm.data.resolve_model_data_config(model)
transforms  = timm.data.create_transform(**data_config, is_training=False)
model       = model.to(DEVICE)
model.eval()

############################################
# FEATURE EXTRACTION
# Steps 3 & 4: EVA-02 forward pass -> 1000-dim softmax embedding
# Records per-batch times and total wall time
############################################

def extract_features(files, desc=""):
    dataset = ImageDataset(files, transforms)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=False)
    preds       = []
    batch_times = []
    start_total = time.time()

    for batch in tqdm(loader, desc=desc):
        t_batch = time.time()
        with torch.no_grad():
            out = model(batch.to(DEVICE)).softmax(dim=1)   # steps 3 & 4
        preds.append(out.cpu().numpy())
        batch_times.append(time.time() - t_batch)

    total_time = time.time() - start_total
    return np.vstack(preds), batch_times, total_time, len(loader)

############################################
# LOAD PRE-TRAINED XGBOOST MODEL
############################################

print(f"Loading pre-trained XGBoost model from: {XGB_MODEL_PATH}")
from xgboost import Booster, DMatrix

booster = Booster()
booster.load_model(XGB_MODEL_PATH)

print("Extracting TEST features...")

ypred_test, test_batch_times, test_total_time, test_batches = \
    extract_features(test_df["file"].tolist(), desc="test feats")

probs = booster.predict(DMatrix(ypred_test))

############################################
# AUC-ROC  (FIX 1: on test set)
############################################

auc = roc_auc_score(test_df["ground_truth"], probs)

############################################
# THRESHOLD SWEEP  0.30 -> 0.70, step 0.01
############################################

thresholds     = np.arange(0.30, 0.71, 0.01)
sweep          = []
best_f1        = 0.0
best_threshold = 0.5

for t in thresholds:
    pred = (probs >= t).astype(int)
    # FIX 1: test labels
    f1   = f1_score(test_df["ground_truth"],        pred)
    acc  = accuracy_score(test_df["ground_truth"],  pred)
    prec = precision_score(test_df["ground_truth"], pred)
    rec  = recall_score(test_df["ground_truth"],    pred)
    sweep.append({
        "threshold": round(float(t), 2),
        "f1":        float(f1),
        "accuracy":  float(acc),
        "precision": float(prec),
        "recall":    float(rec)
    })
    if f1 > best_f1:
        best_f1        = f1
        best_threshold = t

############################################
# METRICS AT THRESHOLD 0.5  (FIX 1: test)
############################################

pred05     = (probs >= 0.5).astype(int)
metrics_05 = dict(
    f1        = float(f1_score(test_df["ground_truth"],        pred05)),
    accuracy  = float(accuracy_score(test_df["ground_truth"],  pred05)),
    precision = float(precision_score(test_df["ground_truth"], pred05)),
    recall    = float(recall_score(test_df["ground_truth"],    pred05))
)

############################################
# METRICS AT BEST THRESHOLD  (FIX 1: test)
############################################

pred_opt    = (probs >= best_threshold).astype(int)
metrics_opt = dict(
    f1        = float(f1_score(test_df["ground_truth"],        pred_opt)),
    accuracy  = float(accuracy_score(test_df["ground_truth"],  pred_opt)),
    precision = float(precision_score(test_df["ground_truth"], pred_opt)),
    recall    = float(recall_score(test_df["ground_truth"],    pred_opt))
)

############################################
# CONFUSION MATRIX AT BEST THRESHOLD (FIX 1: test)
############################################

tn, fp, fn, tp = confusion_matrix(test_df["ground_truth"], pred_opt).ravel()

############################################
# SAVE evaluation_metrics.json
############################################

evaluation_metrics = dict(
    pipeline_name     = PIPELINE_NAME,
    device            = DEVICE,
    timestamp         = str(datetime.datetime.now(datetime.timezone.utc)),
    total_samples     = int(len(test_df)),           # 5860
    auc_roc           = float(auc),
    threshold_0_5     = metrics_05,
    optimal_threshold = float(best_threshold),
    optimal_metrics   = metrics_opt,
    confusion_matrix  = dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn)),
    threshold_sweep   = sweep
)

eval_path = f"{RESULTS_DIR}/python_cpu_evaluation_metrics.json"
with open(eval_path, "w") as f:
    json.dump(evaluation_metrics, f, indent=4)
print(f"Evaluation metrics saved to: {eval_path}")

# ==========================================================
# BENCHMARKING - FULL PIPELINE RUNS
# Reuses `model`, `booster`, `transforms` already loaded.
# Each profiling run: load PNG -> EVA-02 -> XGBoost -> prediction
# No cached embeddings - full raw image path every run.
# ==========================================================

n_test_images = len(test_df)           # 5860
test_files    = test_df["file"].tolist()

# Shared dataset - transforms already loaded, no re-init
dataset_test  = ImageDataset(test_files, transforms)

# ----------------------------------------------------------
# INFERENCE CLOSURE
# Full raw-image pipeline per call - no cached embeddings.
# Records per-batch times, peak CPU RAM.
#
# include_xgb=False -> Pipeline 1: PNG -> EVA-02 -> embedding
# include_xgb=True  -> Pipeline 2: PNG -> EVA-02 -> XGBoost -> prediction
# ----------------------------------------------------------

def make_pipeline_fn(include_xgb):
    def inference_fn():
        loader = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=False)
        batch_times = []

        for batch in loader:
            t_b = time.time()
            with torch.no_grad():
                feats = model(batch.to(DEVICE)).softmax(dim=1)  # steps 3 & 4
            if include_xgb:
                _ = booster.predict(DMatrix(feats.cpu().numpy()))
            batch_times.append(time.time() - t_b)

        peak_cpu  = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

        return dict(
            status           = "done",
            n_images         = n_test_images,
            batch_times      = batch_times,
            peak_gpu_vram_mb = 0.0,
            peak_cpu_ram_mb  = peak_cpu,
        )
    return inference_fn

# ==========================================================
# PIPELINE 1 - EVA-02 only
# ==========================================================

print("\n" + "-" * 65)
print("PIPELINE 1 - EVA-02 feature extraction only")
print("-" * 65)

# Warm-up: 1 FULL PASS through ALL 5860 test images (not measured)
print("\n[WARM-UP P1] Pass 1/2 (not measured)...")
p1_fn = make_pipeline_fn(include_xgb=False)
_ = p1_fn()
print("[WARM-UP P1] Pass 2/2 - stable state (not measured)...")
_ = p1_fn()
print("   Warm-up complete.")

# 5 measured runs, 30 s cooldown between each
run_wall_s_p1      = []
all_batch_times_p1 = []
run_cpu_ram_p1     = []

print(f"\n Running {N_PROFILE_RUNS} measured inference runs ...")
print(f"   ({COOLDOWN_S:.0f}s cool-down between runs)")

for r in range(1, N_PROFILE_RUNS + 1):
    print(f"   Run {r}/{N_PROFILE_RUNS} - ", end="", flush=True)

    t0     = time.time()
    result = p1_fn()
    t1     = time.time()
    wall   = t1 - t0

    run_wall_s_p1.append(wall)
    all_batch_times_p1.append(result["batch_times"])
    run_cpu_ram_p1.append(result["peak_cpu_ram_mb"])

    print(f"wall {wall:.2f}s  peak RAM {result['peak_cpu_ram_mb']:.1f} MB")

    if r < N_PROFILE_RUNS:
        print(f"   Cooling down {COOLDOWN_S:.0f}s ...")
        time.sleep(COOLDOWN_S)

# ==========================================================
# PIPELINE 2 - EVA-02 + XGBoost (full combined pipeline)
# ==========================================================

print("\n" + "-" * 65)
print("PIPELINE 2 - EVA-02 + XGBoost (combined, full pipeline)")
print("-" * 65)

# Warm-up: 1 FULL PASS through ALL 5860 test images (not measured)
print("\n[WARM-UP P2] Pass 1/2 (not measured)...")
p2_fn = make_pipeline_fn(include_xgb=True)
_ = p2_fn()
print("[WARM-UP P2] Pass 2/2 - stable state (not measured)...")
_ = p2_fn()
print("   Warm-up complete.")

# 5 measured runs, 30 s cooldown between each
run_wall_s_p2      = []
all_batch_times_p2 = []
run_cpu_ram_p2     = []

print(f"\n Running {N_PROFILE_RUNS} measured inference runs ...")
print(f"   ({COOLDOWN_S:.0f}s cool-down between runs)")

for r in range(1, N_PROFILE_RUNS + 1):
    print(f"   Run {r}/{N_PROFILE_RUNS} - ", end="", flush=True)

    t0     = time.time()
    result = p2_fn()
    t1     = time.time()
    wall   = t1 - t0

    run_wall_s_p2.append(wall)
    all_batch_times_p2.append(result["batch_times"])
    run_cpu_ram_p2.append(result["peak_cpu_ram_mb"])

    print(f"wall {wall:.2f}s  peak RAM {result['peak_cpu_ram_mb']:.1f} MB")

    if r < N_PROFILE_RUNS:
        print(f"   Cooling down {COOLDOWN_S:.0f}s ...")
        time.sleep(COOLDOWN_S)

# ==========================================================
# SAVE python_cpu_inference_metrics.json
# Derived from the 5 measured runs of Pipeline 2 (full pipeline).
# All individual batch times from every run included.
# Mirrors GPU script's python_gpu_inference_metrics.json
# structure exactly — same field names, same derivations.
# ==========================================================

def safe_mean(v): return float(np.mean(v))        if len(v) > 0 else float("nan")
def safe_std(v):  return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")

all_batch_times_flat = [t for run in all_batch_times_p2 for t in run]
per_run_total_s      = run_wall_s_p2
n_valid_runs         = len(per_run_total_s)

print(f"\n   Aggregate Pipeline 2 ({n_valid_runs} runs): "
      f"{safe_mean(per_run_total_s):.2f} +/- {safe_std(per_run_total_s):.2f} s")

inference_metrics = dict(
    pipeline_name                 = PIPELINE_NAME,
    device                        = DEVICE,
    timestamp                     = str(datetime.datetime.now(datetime.timezone.utc)),
    batch_size                    = BATCH_SIZE,
    total_images                  = n_test_images,
    total_batches                 = int(np.ceil(n_test_images / BATCH_SIZE)),
    n_measured_runs               = n_valid_runs,
    throughput_images_per_sec     = float(n_test_images / safe_mean(per_run_total_s)),
    latency_ms_per_image          = float(safe_mean(per_run_total_s) / n_test_images * 1000),
    total_inference_time_sec_mean = float(safe_mean(per_run_total_s)),
    total_inference_time_sec_std  = float(safe_std(per_run_total_s)),
    total_inference_time_per_run  = [round(v, 4) for v in per_run_total_s],
    batch_times_all_runs          = all_batch_times_p2,
    batch_time_mean_s             = float(safe_mean(all_batch_times_flat)),
    batch_time_std_s              = float(safe_std(all_batch_times_flat)),
    peak_gpu_vram_mb              = 0.0,
    peak_cpu_ram_mb               = float(safe_mean(run_cpu_ram_p2))
                                    if run_cpu_ram_p2 else 0.0,
)

inference_path = f"{RESULTS_DIR}/python_cpu_inference_metrics.json"
with open(inference_path, "w") as f:
    json.dump(inference_metrics, f, indent=4)
print(f"Inference metrics saved to: {inference_path}")

# ==========================================================
# PRINT PUBLISHABLE SUMMARIES
# ==========================================================

print(f"\n{'=' * 65}")
print(f"  Inference (Pipeline 2 - {n_valid_runs}-run mean)")
print(f"{'=' * 65}")
print(f"   Throughput    : {inference_metrics['throughput_images_per_sec']:.1f} img/s")
print(f"   Latency       : {inference_metrics['latency_ms_per_image']:.3f} ms/img")
print(f"   Wall time     : {inference_metrics['total_inference_time_sec_mean']:.2f} +/- "
      f"{inference_metrics['total_inference_time_sec_std']:.2f} s")
print(f"   Peak CPU RAM  : {inference_metrics['peak_cpu_ram_mb']:.1f} MB")
print(f"   Batch size    : {inference_metrics['batch_size']}")
print(f"   Total images  : {inference_metrics['total_images']}")
print(f"   Total batches : {inference_metrics['total_batches']}")
print("=" * 65)
print("\nDone.")

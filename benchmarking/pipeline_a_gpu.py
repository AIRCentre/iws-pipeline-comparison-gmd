# =========================================================
# EVA02 + XGBoost — Ocean Internal Wave Classification
# WITH GPU POWER PROFILING (nvidia-smi synchronous polling)
#
# Author:  Gilberto Titericz (pipeline design)
#          Arun Shukla (benchmarking integration, power profiling)
# Revised: João Pinelo (removed unused EVA02-only profiling run,
#          quality assurance)
# =========================================================
# OUTPUT FILES (as written on server; renamed to pipeline_a_* in repo):
#   • python_gpu_evaluation_metrics.json  → pipeline_a_gpu_evaluation.json
#   • python_gpu_inference_metrics.json   → pipeline_a_gpu_inference.json
#   • python_gpu_power_profile.json       → pipeline_a_gpu_power.json
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
#   • Warm-up        : 2 FULL PASSES through all 5860 test images (not measured)
#   • Profiling runs : 5 measured runs (mean ± std reported)
#   • Power polling  : 100 ms nvidia-smi intervals, inference only
#   • Idle baseline  : 60 s per pipeline (separate for P1 and P2)
#   • Cooldown       : 30 s between consecutive runs
#   • Excluded       : model load, data preload, warm-up pass
# =========================================================

import os
import subprocess
import threading
import time
import json
import datetime
import traceback
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
RESULTS_DIR   = f"{BASE_DIR}/gpu_python_iws/results"

TEST_IMG_DIR   = f"{DATA_DIR}/test"
TEST_CSV       = f"{DATA_DIR}/test.csv"

XGB_MODEL_PATH = "/tmp/benchmarks_iws/gpu_python_iws/results/xgboost_model.json"

OUTPUT_POWER_JSON = f"{RESULTS_DIR}/python_gpu_power_profile.json"

BATCH_SIZE    = 128
NUM_WORKERS   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINE_NAME = "eva02_xgboost_iws"

# Power profiling settings
NVIDIA_SMI          = "/usr/bin/nvidia-smi"
IDLE_DURATION_S     = 60.0      # 60 s idle baseline per pipeline
INTERVAL_MS         = 100       # 100 ms nvidia-smi polling
DISCARD_START_S     = 2.0
DISCARD_END_S       = 2.0
COOLDOWN_S          = 30.0      # 30 s cooldown between consecutive runs
N_PROFILE_RUNS      = 5         # 5 measured runs
THERMAL_THRESH_C    = 80.0

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
if hasattr(torch, "compile"):
    print("Compiling EVA-02 model (torch.compile)...")
    model = torch.compile(model, mode="default")

############################################
# FEATURE EXTRACTION
# Steps 3 & 4: EVA-02 forward pass -> 1000-dim softmax embedding
# Records per-batch times and total wall time
############################################

def extract_features(files, desc=""):
    dataset = ImageDataset(files, transforms)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=(DEVICE == "cuda"))
    preds       = []
    batch_times = []
    start_total = time.time()

    for batch in tqdm(loader, desc=desc):
        t_batch = time.time()
        with torch.no_grad():
            out = model(batch.to(DEVICE)).softmax(dim=1)   # steps 3 & 4
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        preds.append(out.cpu().numpy())
        batch_times.append(time.time() - t_batch)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
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
if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()

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

eval_path = f"{RESULTS_DIR}/python_gpu_evaluation_metrics.json"
with open(eval_path, "w") as f:
    json.dump(evaluation_metrics, f, indent=4)
print(f"Evaluation metrics saved to: {eval_path}")

# ==========================================================
# POWER PROFILER FUNCTIONS
# ==========================================================

def _smi_query(field):
    r = subprocess.run(
        [NVIDIA_SMI, f"--query-gpu={field}",
         "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True, timeout=5
    )
    return r.stdout.strip()

def gpu_metadata():
    meta = {"measurement_date": str(datetime.datetime.now()),
            "nvidia_smi_path": NVIDIA_SMI}
    try:
        meta["gpu_name"]       = _smi_query("name")
        meta["driver_version"] = _smi_query("driver_version")
        for key, field in [("tdp_w",              "power.limit"),
                            ("vram_total_mb",      "memory.total"),
                            ("graphics_clock_mhz", "clocks.gr"),
                            ("memory_clock_mhz",   "clocks.mem")]:
            try:    meta[key] = float(_smi_query(field))
            except: meta[key] = None
        meta["persistence_mode"] = _smi_query("persistence_mode")
        # FIX 5: clocks_locked correctly set to false
        meta["clocks_locked"] = False
    except Exception as e:
        meta["metadata_error"] = str(e)
    return meta

def sample_gpu_once():
    try:
        r = subprocess.run(
            [NVIDIA_SMI, "--query-gpu=power.draw,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=3
        )
        parts = r.stdout.strip().split(",")
        if len(parts) < 2:
            return (float("nan"), float("nan"))
        return (float(parts[0].strip()), float(parts[1].strip()))
    except Exception:
        return (float("nan"), float("nan"))

class NvidiaSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms  = interval_ms
        self.power_w      = []
        self.temp_c       = []
        self.timestamps   = []
        self.running      = False
        self._thread      = None
        self._t_start     = 0.0
        self.sample_count = 0

    def start(self):
        self.power_w = []; self.temp_c = []; self.timestamps = []
        self.running  = True
        self._t_start = time.time()
        self.sample_count = 0
        self._thread = threading.Thread(target=self._safe_loop, daemon=True)
        self._thread.start()

    def _safe_loop(self):
        try:
            self._loop()
        except Exception as e:
            print(f"\n   Sampler thread crashed: {e}")
            traceback.print_exc()
            self.running = False

    def _loop(self):
        while self.running:
            try:
                pw, tp = sample_gpu_once()
                if pw == pw:
                    self.power_w.append(pw)
                    self.temp_c.append(tp)
                    self.timestamps.append(time.time() - self._t_start)
                    self.sample_count += 1
            except Exception as e:
                print(f"   Sampler error: {e}")
            time.sleep(self.interval_ms / 1000.0)

    def stop(self):
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)

def safe_mean(v): return float(np.mean(v))         if len(v) > 0 else float("nan")
def safe_std(v):  return float(np.std(v, ddof=1))  if len(v) > 1 else float("nan")
def safe_min(v):  return float(np.min(v))           if len(v) > 0 else float("nan")
def safe_max(v):  return float(np.max(v))           if len(v) > 0 else float("nan")

def iqr_trim(v):
    """IQR-robust Tukey fences - identical logic to Julia iqr_trim().
    Removes outlier power spikes before computing mean.
    Keeps samples within [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    if len(v) < 4:
        return v
    arr = np.array(v)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    trimmed = arr[(arr >= lo) & (arr <= hi)].tolist()
    return trimmed if trimmed else v

def trim_samples(power_w, temp_c, timestamps,
                 discard_start_s=0.0, discard_end_s=0.0):
    if not timestamps:
        return power_w, temp_c, timestamps
    total = timestamps[-1]
    if total <= (discard_start_s + discard_end_s + 0.5):
        return power_w, temp_c, timestamps
    lo, hi = discard_start_s, total - discard_end_s
    mask = [lo <= t <= hi for t in timestamps]
    if not any(mask):
        return power_w, temp_c, timestamps
    return (
        [p for p, m in zip(power_w,    mask) if m],
        [t for t, m in zip(temp_c,     mask) if m],
        [t for t, m in zip(timestamps, mask) if m],
    )

def check_thermal_throttling(temp_c, threshold_c=THERMAL_THRESH_C):
    if not temp_c:
        return dict(throttled=False, max_temp=float("nan"),
                    mean_temp=float("nan"), pct_above=0.0)
    pct = 100.0 * sum(t > threshold_c for t in temp_c) / len(temp_c)
    return dict(throttled  = safe_max(temp_c) > threshold_c,
                max_temp   = safe_max(temp_c),
                mean_temp  = safe_mean(temp_c),
                pct_above  = pct)

def measure_idle_baseline(duration_s=IDLE_DURATION_S,
                           interval_ms=INTERVAL_MS,
                           discard_start_s=DISCARD_START_S,
                           discard_end_s=DISCARD_END_S):
    print(f"\n Measuring GPU idle baseline for {duration_s:.0f}s ...")
    print("   (GPU must be idle - no other GPU workloads running)")
    s = NvidiaSampler(interval_ms=interval_ms)
    s.start()
    time.sleep(duration_s)
    s.stop()
    print(f"   Raw samples collected: {s.sample_count}")

    pw, tp, _ = trim_samples(s.power_w, s.temp_c, s.timestamps,
                              discard_start_s, discard_end_s)
    if not pw:
        pw, tp = s.power_w, s.temp_c
    if not pw:
        raise RuntimeError("No idle samples - check nvidia-smi")

    # Convergence check — identical logic to Julia Section 8
    converged = True
    convergence_note = "OK"
    if len(pw) >= 9:
        third = len(pw) // 3
        w1 = float(np.mean(pw[:third]))
        w2 = float(np.mean(pw[third:2*third]))
        w3 = float(np.mean(pw[2*third:]))
        drift = max(w1, w2, w3) - min(w1, w2, w3)
        if drift > 1.0:
            converged = False
            convergence_note = f"Drift {drift:.2f} W across windows - extend idle duration"
            print(f"   WARNING: Idle not converged (drift={drift:.2f} W)")
        else:
            convergence_note = f"Converged (drift {drift:.2f} W across 3 windows)"
    print(f"   Convergence: {convergence_note}")

    result = dict(
        mean_w           = safe_mean(pw),
        std_w            = safe_std(pw),
        min_w            = safe_min(pw),
        max_w            = safe_max(pw),
        mean_temp_c      = safe_mean(tp),
        min_temp_c       = safe_min(tp),
        max_temp_c       = safe_max(tp),
        duration_s       = duration_s,
        n_samples_total  = s.sample_count,
        n_samples_used   = len(pw),
        converged        = converged,
        convergence_note = convergence_note,
    )
    print(f"   Idle power : {result['mean_w']:.2f} +/- {result['std_w']:.2f} W"
          f"  (min {result['min_w']:.2f}, max {result['max_w']:.2f})")
    print(f"   Idle temp  : {result['mean_temp_c']:.1f} C")
    return result

def measure_one_run(inference_fn, interval_ms=INTERVAL_MS,
                    discard_start_s=0.0, discard_end_s=0.0,
                    idle=None):
    s = NvidiaSampler(interval_ms=interval_ms)
    s.start()
    time.sleep(0.5)          # allow sampler thread to fire at least once before inference starts
    t0     = time.time()
    result = inference_fn()
    t1     = time.time()
    s.stop()
    wall = t1 - t0

    print(f"[sampler: {s.sample_count} raw samples, wall {wall:.2f}s] ", end="")
    if s.sample_count < 10:
        print(f"\n     WARNING: Only {s.sample_count} samples!")

    pw, tp, _ = trim_samples(s.power_w, s.temp_c, s.timestamps,
                              discard_start_s, discard_end_s)
    if not pw:
        pw = s.power_w or [0.0]
        tp = s.temp_c  or [0.0]

    pw_robust = iqr_trim(pw)
    mean_w    = safe_mean(pw_robust)
    peak_w    = safe_max(pw)
    p95_w     = float(np.percentile(pw, 95)) if len(pw) > 0 else float("nan")

    energy_j = mean_w * wall
    net_w    = float("nan") if idle is None else mean_w - idle["mean_w"]
    thermal  = check_thermal_throttling(tp)

    return dict(
        mean_w            = mean_w,
        std_w             = safe_std(pw_robust),
        peak_w            = peak_w,
        p95_w             = p95_w,
        net_w             = net_w,
        energy_j          = energy_j,
        wall_time_s       = wall,
        mean_temp_c       = safe_mean(tp),
        max_temp_c        = safe_max(tp),
        throttled         = thermal["throttled"],
        pct_above_80c     = thermal["pct_above"],
        n_samples_trimmed = len(pw),
        n_samples_total   = s.sample_count,
        inference_result  = result,
        batch_times       = result.get("batch_times", []),
        peak_gpu_vram_mb  = result.get("peak_gpu_vram_mb", 0.0),
        peak_cpu_ram_mb   = result.get("peak_cpu_ram_mb",  0.0),
    )

def measure_power_five_runs(inference_fn, interval_ms=INTERVAL_MS,
                             discard_start_s=0.0, discard_end_s=0.0,
                             cooldown_s=COOLDOWN_S, idle=None):
    run_mean_w    = []; run_peak_w   = []; run_p95_w    = []; run_net_w    = []
    run_energy_j  = []; run_wall_s  = []; run_temp_c   = []
    run_throttled = []; run_samples = []; failed_runs  = []
    all_batch_times = []
    run_gpu_vram  = []; run_cpu_ram = []
    print(f"\n Running {N_PROFILE_RUNS} power-profiled inference runs ...")
    print(f"   ({cooldown_s:.0f}s cool-down between runs)")

    for r in range(1, N_PROFILE_RUNS + 1):
        print(f"   Run {r}/{N_PROFILE_RUNS} - ", end="", flush=True)
        m = None
        for attempt in range(2):
            m = measure_one_run(inference_fn,
                                interval_ms=interval_ms,
                                discard_start_s=discard_start_s,
                                discard_end_s=discard_end_s,
                                idle=idle)
            if m["n_samples_total"] >= 10:
                break
            print(f"\n     Attempt {attempt+1}: only {m['n_samples_total']} samples, retrying...")
            time.sleep(5)

        if m["n_samples_total"] < 10:
            print(f"\n     Run {r} failed - excluded from aggregates")
            failed_runs.append(r)
        else:
            run_mean_w.append(m["mean_w"])
            run_peak_w.append(m["peak_w"])
            run_p95_w.append(m["p95_w"])
            run_net_w.append(m["net_w"])
            run_energy_j.append(m["energy_j"])
            run_wall_s.append(m["wall_time_s"])
            run_temp_c.append(m["mean_temp_c"])
            run_throttled.append(m["throttled"])
            run_samples.append(m["n_samples_total"])
            all_batch_times.append(m["batch_times"])
            run_gpu_vram.append(m["peak_gpu_vram_mb"])
            run_cpu_ram.append(m["peak_cpu_ram_mb"])

        net_str = "N/A" if m["net_w"] != m["net_w"] else f"{m['net_w']:.2f}"
        flag    = " THROTTLE" if m["throttled"] else ""
        print(f"{m['mean_w']:.2f} W (net {net_str} W)  "
              f"peak {m['peak_w']:.2f} W  "
              f"{m['mean_temp_c']:.1f}C{flag}  "
              f"{m['wall_time_s']:.2f}s")

        if r < N_PROFILE_RUNS:
            print(f"   Cooling down {cooldown_s:.0f}s ...")
            time.sleep(cooldown_s)

    if failed_runs:
        print(f"\n   WARNING: Excluded runs (insufficient samples): {failed_runs}")
    if run_throttled and any(run_throttled):
        print("   WARNING: Thermal throttling detected in >=1 run!")

    valid_net  = [v for v in run_net_w if v == v]
    mean_w_agg = safe_mean(run_mean_w)
    std_w_agg  = safe_std(run_mean_w)
    net_w_agg  = safe_mean(valid_net) if valid_net else float("nan")

    if run_mean_w:
        print(f"\n   Aggregate ({len(run_mean_w)} valid runs): "
              f"{mean_w_agg:.2f} +/- {std_w_agg:.2f} W  "
              f"peak {safe_mean(run_peak_w):.2f} W")

    return dict(
        run_mean_w      = run_mean_w,
        run_peak_w      = run_peak_w,
        run_p95_w       = run_p95_w,
        run_net_w       = run_net_w,
        run_energy_j    = run_energy_j,
        run_wall_s      = run_wall_s,
        run_temp_c      = run_temp_c,
        run_throttled   = run_throttled,
        run_samples     = run_samples,
        failed_runs     = failed_runs,
        all_batch_times = all_batch_times,
        run_gpu_vram    = run_gpu_vram,
        run_cpu_ram     = run_cpu_ram,
        mean_w          = mean_w_agg,
        std_w           = std_w_agg,
        mean_peak_w     = safe_mean(run_peak_w)   if run_peak_w   else float("nan"),
        mean_p95_w      = safe_mean(run_p95_w),
        mean_energy_j   = safe_mean(run_energy_j) if run_energy_j else float("nan"),
        net_w           = net_w_agg,
        any_throttled   = any(run_throttled) if run_throttled else False,
    )

def compute_energy_metrics(power, n_images, idle=None):
    mean_w    = power["mean_w"]
    net_w     = power["net_w"]
    mean_wall = safe_mean(power["run_wall_s"]) if power["run_wall_s"] else float("nan")
    std_wall  = safe_std(power["run_wall_s"])  if len(power["run_wall_s"]) > 1 else float("nan")

    gross_energy_j   = mean_w * mean_wall
    gross_per_img_mj = (gross_energy_j / n_images) * 1000
    net_per_img_mj   = None if (net_w != net_w) else (net_w * mean_wall / n_images) * 1000
    energy_j_list    = power["run_energy_j"]

    return dict(
        gross_mean_w              = mean_w,
        gross_std_w               = power["std_w"],
        net_w                     = None if (net_w != net_w) else net_w,
        idle_baseline_w           = None if idle is None else idle["mean_w"],
        mean_wall_time_s          = mean_wall,
        std_wall_time_s           = std_wall,
        gross_total_energy_j      = gross_energy_j,
        gross_energy_per_image_j  = gross_energy_j / n_images,
        gross_energy_per_image_mj = gross_per_img_mj,
        net_energy_per_image_mj   = net_per_img_mj,
        mean_energy_j_per_run     = safe_mean(energy_j_list),
        std_energy_j_per_run      = safe_std(energy_j_list),
    )

def build_power_report(pipeline_name, n_images, batch_size,
                       idle, power, extra_fields=None):
    energy   = compute_energy_metrics(power, n_images, idle)
    gpu_meta = gpu_metadata()

    per_run = []
    for i in range(len(power["run_mean_w"])):
        nv = power["run_net_w"][i]
        per_run.append(dict(
            run         = i + 1,
            mean_w      = round(power["run_mean_w"][i],  3),
            peak_w      = round(power["run_peak_w"][i],  3),
            p95_w       = round(power["run_p95_w"][i],   3),
            net_w       = None if (nv != nv) else round(nv, 3),
            energy_j    = round(power["run_energy_j"][i],3),
            wall_time_s = round(power["run_wall_s"][i],  3),
            mean_temp_c = round(power["run_temp_c"][i],  2),
            throttled   = power["run_throttled"][i],
            n_samples   = power["run_samples"][i],
        ))

    idle_block = dict(
        mean_w           = round(idle["mean_w"],       3),
        std_w            = round(idle["std_w"],        3),
        min_w            = round(idle["min_w"],        3),
        max_w            = round(idle["max_w"],        3),
        mean_temp_c      = round(idle["mean_temp_c"],  2),
        min_temp_c       = round(idle["min_temp_c"],   2),
        max_temp_c       = round(idle["max_temp_c"],   2),
        duration_s       = idle["duration_s"],
        n_samples_total  = idle["n_samples_total"],
        n_samples_used   = idle["n_samples_used"],
        converged        = idle.get("converged", True),
        convergence_note = idle.get("convergence_note", "OK"),
    )

    nwv = power["net_w"]
    cross_run = dict(
        power_mean_w         = round(power["mean_w"],                   3),
        power_std_w          = round(power["std_w"],                    3),
        power_min_w          = round(safe_min(power["run_mean_w"]),     3),
        power_max_w          = round(safe_max(power["run_mean_w"]),     3),
        peak_power_mean_w    = round(power["mean_peak_w"],              3),
        net_power_mean_w     = None if (nwv != nwv) else round(nwv,    3),
        energy_mean_j        = round(energy["mean_energy_j_per_run"],   3),
        energy_std_j         = round(energy["std_energy_j_per_run"],    3),
        wall_time_mean_s     = round(energy["mean_wall_time_s"],        3),
        wall_time_std_s      = round(energy["std_wall_time_s"],         3),
        valid_runs           = len(power["run_mean_w"]),
        failed_runs          = power["failed_runs"],
        any_thermal_throttle = power["any_throttled"],
    )

    report = dict(
        metadata = dict(
            pipeline                 = pipeline_name,
            generated_at             = str(datetime.datetime.now()),
            measurement_tool         = "nvidia-smi synchronous polling",
            sampling_interval_ms     = INTERVAL_MS,
            n_profile_runs           = N_PROFILE_RUNS,
            cooldown_between_runs_s  = COOLDOWN_S,
            idle_baseline_duration_s = IDLE_DURATION_S,
            thermal_threshold_c      = THERMAL_THRESH_C,
            warmup_protocol          = "2 warm-up passes through all test images before measurement (not measured)",
            notes = [
                "Full pipeline per run: load PNG -> resize 448x448 -> EVA-02 -> 1000-dim embedding -> XGBoost",
                "Power measured during inference loop only - model load, preload, warm-up excluded",
                "Net power = gross mean power - idle baseline mean power",
                "gross_energy_per_image_mj = (mean_w x mean_wall_s / n_images) x 1000",
                "net_energy_per_image_mj   = (net_w  x mean_wall_s / n_images) x 1000",
                f"Cross-run stats = mean +/- std across {len(power['run_mean_w'])} valid runs",
            ],
        ),
        system            = dict(gpu=gpu_meta, n_images=n_images, batch_size=batch_size),
        idle_baseline     = idle_block,
        per_run_breakdown = per_run,
        energy_metrics    = dict(
            gross_energy_per_image_mj = round(energy["gross_energy_per_image_mj"], 4)
                                        if energy["gross_energy_per_image_mj"] == energy["gross_energy_per_image_mj"] else None,
            net_energy_per_image_mj   = None if energy["net_energy_per_image_mj"] is None
                                        else round(energy["net_energy_per_image_mj"], 4),
            gross_total_energy_j      = round(energy["gross_total_energy_j"],     3),
            gross_energy_per_image_j  = round(energy["gross_energy_per_image_j"], 6),
            idle_baseline_w           = energy["idle_baseline_w"],
            mean_wall_time_s          = round(energy["mean_wall_time_s"],          3),
        ),
        cross_run_stats   = cross_run,
        publishable_summary = dict(
            mean_gpu_power_w          = round(power["mean_w"], 2) if power["mean_w"] == power["mean_w"] else None,
            std_gpu_power_w           = round(power["std_w"],  2) if power["std_w"]  == power["std_w"]  else None,
            net_inference_power_w     = None if (nwv != nwv) else round(nwv, 2),
            energy_per_image_mj_gross = round(energy["gross_energy_per_image_mj"], 3)
                                        if energy["gross_energy_per_image_mj"] == energy["gross_energy_per_image_mj"] else None,
            std_energy_per_image_mj   = round(energy["std_energy_j_per_run"] / n_images * 1000, 3),
            energy_per_image_mj_net   = None if energy["net_energy_per_image_mj"] is None
                                        else round(energy["net_energy_per_image_mj"], 3),
            thermal_throttling        = power["any_throttled"],
            report_format             = f"mean +/- std across {len(power['run_mean_w'])} valid runs",
        ),
    )
    if extra_fields:
        report.update(extra_fields)
    return report

# ==========================================================
# POWER PROFILING - FULL PIPELINE RUNS
# Reuses `model`, `model_xgb`, `transforms` already loaded.
# Each profiling run: load PNG -> EVA-02 -> XGBoost -> prediction
# No cached embeddings - full raw image path every run.
# ==========================================================

print("\n" + "=" * 65)
print("GPU POWER PROFILING - EVA02 + XGBoost (IWS)")
print("=" * 65)

if not (os.path.isfile(NVIDIA_SMI) and os.access(NVIDIA_SMI, os.X_OK)):
    raise RuntimeError(f"nvidia-smi not found at {NVIDIA_SMI}")

n_test_images = len(test_df)           # 5860
test_files    = test_df["file"].tolist()

# Shared dataset - transforms already loaded, no re-init
dataset_test  = ImageDataset(test_files, transforms)

# ----------------------------------------------------------
# INFERENCE CLOSURE
# Full raw-image pipeline per call - no cached embeddings.
# Records per-batch times, peak VRAM, peak CPU RAM.
#
# include_xgb=True  -> Full pipeline: PNG -> EVA-02 -> XGBoost -> prediction
# ----------------------------------------------------------

def make_pipeline_fn(include_xgb):
    def inference_fn():
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        loader = DataLoader(dataset_test, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=(DEVICE == "cuda"))
        batch_times = []

        for batch in loader:
            t_b = time.time()
            with torch.no_grad():
                feats = model(batch.to(DEVICE)).softmax(dim=1)  # steps 3 & 4
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            if include_xgb:
                _ = booster.predict(DMatrix(feats.cpu().numpy()))
            batch_times.append(time.time() - t_b)

        peak_vram = torch.cuda.max_memory_allocated() / (1024*1024) \
                    if DEVICE == "cuda" else 0.0
        peak_cpu  = psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

        return dict(
            status           = "done",
            n_images         = n_test_images,
            batch_times      = batch_times,
            peak_gpu_vram_mb = peak_vram,
            peak_cpu_ram_mb  = peak_cpu,
        )
    return inference_fn

# ==========================================================
# EVA-02 + XGBoost (full combined pipeline)
# ==========================================================

print("\n" + "-" * 65)
print("EVA-02 + XGBoost — Full combined pipeline")
print("-" * 65)

# Idle baseline - 60 s, separate measurement for Pipeline 2
print("\nCooling GPU before idle baseline (120s) ...")
time.sleep(120)
print("\n--- IDLE BASELINE (Pipeline 2, 60s) ---")
idle2 = measure_idle_baseline()
# Warm-up: 1 FULL PASS through ALL 5860 test images (not measured)
print("\n[WARM-UP P2] Pass 1/2 - triggers torch.compile (not measured)...")
p2_fn = make_pipeline_fn(include_xgb=True)
_ = p2_fn()
print("[WARM-UP P2] Pass 2/2 - fully compiled stable state (not measured)...")
_ = p2_fn()
print("   Warm-up complete.")
# 5 measured runs, 30 s cooldown between each
power2 = measure_power_five_runs(
    p2_fn,
    interval_ms     = INTERVAL_MS,
    discard_start_s = DISCARD_START_S,
    discard_end_s   = DISCARD_END_S,
    cooldown_s      = COOLDOWN_S,
    idle            = idle2,
)
report2 = build_power_report(
    pipeline_name = "EVA02_XGBoost_combined_GPU",
    n_images      = n_test_images,
    batch_size    = BATCH_SIZE,
    idle          = idle2,
    power         = power2,
)

# ==========================================================
# SAVE python_gpu_power_profile.json
# ==========================================================

final_power = dict(
    metadata = dict(
        description              = "GPU power profiling - EVA02 + XGBoost IWS pipeline",
        generated_at             = str(datetime.datetime.now()),
        measurement_tool         = "nvidia-smi synchronous polling",
        sampling_interval_ms     = INTERVAL_MS,
        idle_baseline_duration_s = IDLE_DURATION_S,
        n_profile_runs           = N_PROFILE_RUNS,
        cooldown_between_runs_s  = COOLDOWN_S,
        warmup                   = "2 warm-up passes through all test images before measurement (not measured)",
        n_test_images            = n_test_images,
        batch_size               = BATCH_SIZE,
    ),
    pipeline_2_eva02_xgboost = report2,
)

with open(OUTPUT_POWER_JSON, "w") as f:
    json.dump(final_power, f, indent=2, default=str)
print(f"\nPower profile saved to: {OUTPUT_POWER_JSON}")

# ==========================================================
# SAVE python_gpu_inference_metrics.json
# Derived from the 5 measured runs of Pipeline 2 (full pipeline).
# All individual batch times from every run included.
# ==========================================================

all_batch_times_flat = [t for run in power2["all_batch_times"] for t in run]
per_run_total_s      = power2["run_wall_s"]
n_valid_runs         = len(per_run_total_s)

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
    batch_times_all_runs          = power2["all_batch_times"],
    batch_time_mean_s             = float(safe_mean(all_batch_times_flat)),
    batch_time_std_s              = float(safe_std(all_batch_times_flat)),
    peak_gpu_vram_mb              = float(safe_mean(power2["run_gpu_vram"]))
                                    if power2["run_gpu_vram"] else 0.0,
    peak_cpu_ram_mb               = float(safe_mean(power2["run_cpu_ram"]))
                                    if power2["run_cpu_ram"] else 0.0,
)

inference_path = f"{RESULTS_DIR}/python_gpu_inference_metrics.json"
with open(inference_path, "w") as f:
    json.dump(inference_metrics, f, indent=4)
print(f"Inference metrics saved to: {inference_path}")

# ==========================================================
# PRINT PUBLISHABLE SUMMARIES
# ==========================================================

s = report2["publishable_summary"]
print(f"\n{'=' * 65}")
print(f"  EVA-02 + XGBoost (combined pipeline)")
print(f"{'=' * 65}")
print(f"   Gross GPU power       : {s['mean_gpu_power_w']} +/- {s['std_gpu_power_w']} W")
print(f"   Net inference power   : {s['net_inference_power_w']} W")
print(f"   Energy/image (gross)  : {s['energy_per_image_mj_gross']} mJ")
print(f"   Energy/image (net)    : {s['energy_per_image_mj_net']} mJ")
if s["thermal_throttling"]:
    print("   WARNING: Thermal throttling detected!")

print(f"\n{'=' * 65}")
print(f"  Inference (Pipeline 2 - {n_valid_runs}-run mean)")
print(f"{'=' * 65}")
print(f"   Throughput    : {inference_metrics['throughput_images_per_sec']:.1f} img/s")
print(f"   Latency       : {inference_metrics['latency_ms_per_image']:.3f} ms/img")
print(f"   Wall time     : {inference_metrics['total_inference_time_sec_mean']:.2f} +/- "
      f"{inference_metrics['total_inference_time_sec_std']:.2f} s")
print(f"   Peak GPU VRAM : {inference_metrics['peak_gpu_vram_mb']:.1f} MB")
print(f"   Peak CPU RAM  : {inference_metrics['peak_cpu_ram_mb']:.1f} MB")
print(f"   Batch size    : {inference_metrics['batch_size']}")
print(f"   Total images  : {inference_metrics['total_images']}")
print(f"   Total batches : {inference_metrics['total_batches']}")
print("=" * 65)
print("\nDone.")



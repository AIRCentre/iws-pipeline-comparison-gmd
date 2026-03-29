# Benchmark Data

Raw benchmark results from the comparative evaluation of two ML inference pipelines
for internal solitary wave (ISW) detection in Sentinel-1 SAR imagery.

All benchmarks were conducted on the IWS operational server at the AIR Data Centre,
Terceira, Azores: 2× Intel Xeon Gold 5420+ (56 cores), 502 GB RAM, NVIDIA L40 GPU
(46 GB GDDR6 ECC), Oracle Linux 9.7 (containerised environment) on Rocky Linux 10.1
host, CUDA 13.1, driver 590.48.01.

Test set: 5,860 images (2,930 ISW positives, 2,930 negatives), held out from all training.

## File-to-paper mapping

### Pipeline A (Python EVA02+XGBoost, 305M parameters)

| File | Paper reference | Key values |
|------|----------------|------------|
| `pipeline_a_gpu_evaluation.json` | Table 2 (Pipeline A GPU column) | AUC-ROC 99.29%, F1 96.26%, optimal threshold 0.48 |
| `pipeline_a_gpu_inference.json` | Table 3 (Pipeline A GPU column) | 25.6 img/s, 228.5 ± 1.0 s wall time |
| `pipeline_a_gpu_power.json` | Table 4 (Pipeline A GPU column) | 299.8 ± 1.8 W, 11,690 mJ/img gross, 8,630 mJ/img net |
| `pipeline_a_cpu_evaluation.json` | Table 2 (Pipeline A CPU column) | AUC-ROC 99.29%, F1 96.24%, optimal threshold 0.64 |
| `pipeline_a_cpu_inference.json` | Table 3 (Pipeline A CPU column) | 0.92 img/s, 6,354 ± 23 s wall time |

### Pipeline B (Julia Lux CNN, 283,329 parameters)

| File | Paper reference | Key values |
|------|----------------|------------|
| `pipeline_b_gpu_evaluation.json` | Table 2 (Pipeline B GPU column) | AUC-ROC 98.90%, F1 95.00%, optimal threshold 0.47 |
| `pipeline_b_gpu_inference.json` | Table 3 (Pipeline B GPU column) | 3,396 img/s, 1.7 ± 0.06 s wall time |
| `pipeline_b_gpu_power.json` | Table 4 (Pipeline B GPU column) | 153.0 ± 8.0 W, 43.7 mJ/img gross, 23.6 mJ/img net |
| `pipeline_b_cpu_evaluation.json` | Table 2 (Pipeline B CPU column) | AUC-ROC 98.90%, F1 95.00%, optimal threshold 0.47 |
| `pipeline_b_cpu_inference.json` | Table 3 (Pipeline B CPU column) | 15.2 img/s, 388 ± 34 s wall time |

## Derived operational projections (Section 6)

The operational projections in Tables 5–7 are computed from the per-image throughput
values in the inference JSON files, applied to the IWS design parameters:

- Daily feed: ~4,000 images/day (full Sentinel-1 constellation at operational capacity).
- Archive: estimated up to 17 million images (Sentinel-1 WV mode, 2014–present).
- Reprocessing cadence: R = 2 (twice per year) and R = 4 (quarterly).

## Measurement protocol

- 5 independent runs per configuration; results reported as mean ± std.
- Throughput measured from first batch to last batch (model loading, preloading, and warm-up excluded).
- GPU power sampled via nvidia-smi at 100 ms intervals during inference only.
- 60-second idle baseline recorded before each pipeline.
- GPU clocks not locked — default boost profiles reflect production operating conditions.
- Minimum 30-second cooldown between runs; 5-minute cooldown between pipelines.

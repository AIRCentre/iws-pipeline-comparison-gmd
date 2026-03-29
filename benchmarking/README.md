# Benchmarking Scripts

Scripts used to generate the benchmark results in `data/`. These scripts are provided
for transparency and reproducibility documentation. They were executed on the IWS
operational server at the AIR Data Centre and contain hardcoded absolute paths to that
server's directory structure. To re-run on a different system, adjust the paths in the
CONFIG sections at the top of each script.

## Server environment

- OS: Oracle Linux 9.7 (containerised environment), host Rocky Linux 10.1, kernel 6.12.0-124.28.1.el10_1.x86_64.
- GPU: NVIDIA L40 (46 GB GDDR6 ECC, 300 W TDP), CUDA 13.1, driver 590.48.01.
- CPU: 2× Intel Xeon Gold 5420+ (28 cores each, 56 total, SMT disabled).
- RAM: 502 GB DDR4.
- Container: full access to GPU, all 56 CPU cores, and 502 GB RAM.
- GPU clocks: not locked (default boost, matching production conditions).

## Pipeline A (Python EVA02+XGBoost)

| Script | Output data files | Description |
|--------|-------------------|-------------|
| `pipeline_a_gpu.py` | `pipeline_a_gpu_evaluation.json`, `pipeline_a_gpu_inference.json`, `pipeline_a_gpu_power.json` | GPU inference, evaluation (threshold sweep, AUC-ROC, confusion matrices), and power profiling. Two warm-up passes before measurement. |
| `pipeline_a_cpu.py` | `pipeline_a_cpu_evaluation.json`, `pipeline_a_cpu_inference.json` | CPU inference and evaluation. No power profiling (CPU configurations are not operationally competitive for archive reprocessing). |

### Python dependencies

PyTorch 2.5.1+cu121 with `torch.compile`, timm 1.0.25 (EVA02-Large checkpoint
`eva02_large_patch14_448.mim_m38m_ft_in1k`), XGBoost 3.2.0, scikit-learn, PIL,
numpy, pandas, psutil.

## Pipeline B (Julia Lux CNN)

| Script | Output data files | Description |
|--------|-------------------|-------------|
| `pipeline_b_gpu_inference.jl` | `pipeline_b_gpu_inference.json`, `pipeline_b_gpu_power.json` | GPU inference (5 profiling runs × 5,860 images) with integrated power profiling via `pipeline_b_gpu_power_profiler.jl`. Three warm-up passes before measurement. |
| `pipeline_b_gpu_evaluate.jl` | `pipeline_b_gpu_evaluation.json` | GPU evaluation: threshold sweep, AUC-ROC, confusion matrices. |
| `pipeline_b_gpu_power_profiler.jl` | (called by `pipeline_b_gpu_inference.jl`) | GPU power measurement module: idle baseline, 5-run profiling protocol, IQR-robust statistics, energy-per-image computation. |
| `pipeline_b_cpu_inference.jl` | `pipeline_b_cpu_inference.json` | CPU inference (5 profiling runs). BLAS threads set to all available cores. |
| `pipeline_b_cpu_evaluate.jl` | `pipeline_b_cpu_evaluation.json` | CPU evaluation: threshold sweep, AUC-ROC, confusion matrices. |
| `pipeline_b_model.jl` | — | SAR_CNN v2 model architecture definition (283,329 parameters). Shared by GPU scripts via `include()`; CPU scripts contain an identical inline copy. |
| `pipeline_b_config.jl` | — | Shared constants: `TARGET_SIZE = (256, 256)`, `BATCH_SIZE = 128`, `RNG_SEED = 42`. |

### Julia dependencies

Julia 1.10.5, Lux.jl 1.9.0, CUDA.jl (LuxCUDA), Images.jl, JLD2, CairoMakie
(evaluation scripts only), JSON3.

## Execution order

The benchmarks were executed in the following order on the production server,
with the GPU idle between pipeline runs:

1. `pipeline_b_gpu_inference.jl` — Julia GPU inference + power profiling (2026-03-15).
2. `pipeline_b_gpu_evaluate.jl` — Julia GPU evaluation (2026-03-15).
3. `pipeline_a_gpu.py` — Python GPU inference + evaluation + power profiling (2026-03-16).
4. `pipeline_b_cpu_inference.jl` — Julia CPU inference (2026-03-17).
5. `pipeline_b_cpu_evaluate.jl` — Julia CPU evaluation (2026-03-17).
6. `pipeline_a_cpu.py` — Python CPU inference + evaluation (2026-03-19/20).

## Repository preparation

Scripts in this directory were cleaned for archival clarity before publication.
Unused code was removed; measurement functions and all reported data are unchanged.

## Notes on preprocessing

The Julia GPU and CPU scripts use different numerical ranges for image preprocessing:
the GPU script scales pixel values to [0, 255] before normalisation, while the CPU
script keeps them in [0, 1]. Both produce bit-identical classification results because
every convolutional layer is followed by BatchNorm, which is invariant to linear input
scaling. The normalisation statistics (μ = 0.380, σ = 0.156) are loaded from the model
checkpoint in all scripts.

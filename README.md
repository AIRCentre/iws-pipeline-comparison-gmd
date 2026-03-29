# iws-pipeline-comparison-gmd

Benchmark data, scripts, and figures accompanying the paper:

> **Choosing an operational inference pipeline for internal solitary wave detection in Sentinel-1 SAR imagery: EVA02+XGBoost versus a Julia Lux CNN**
>
> Pinelo, J., Shukla, A., Titericz, G., Santos-Ferreira, A., Gonçalves, J., and Moniz, J.
>
> Submitted to *Geoscientific Model Development* (GMD), 2026.

## Overview

This repository contains the complete benchmark results, benchmarking scripts, trained
model checkpoints, and figure-generation code for a comparative evaluation of two ML
inference pipelines deployed in the [Internal Waves Service](https://www.aircentre.org)
(IWS) — an operational platform for automated detection of internal solitary waves
(ISWs) in Sentinel-1 WV mode SAR imagery, operated by the Atlantic International
Research Centre (AIR Centre).

### Pipelines compared

- **Pipeline A** (Python): EVA02-Large vision transformer (305M parameters) as a
  feature extractor, followed by an XGBoost classifier. Built on PyTorch 2.5 with
  `torch.compile`.
- **Pipeline B** (Julia): SAR_CNN v2, a purpose-built CNN (283,329 parameters)
  implemented in Lux.jl. Trained from scratch on IWS SAR imagery.

### Key results

| Metric | Pipeline A (Python GPU) | Pipeline B (Julia GPU) |
|--------|------------------------|----------------------|
| F1 score | 96.26% | 95.00% |
| AUC-ROC | 99.29% | 98.90% |
| Throughput | 25.6 img/s | 3,396 img/s |
| Energy per image (gross) | 11,690 mJ | 43.7 mJ |
| Archive reprocessing (17M images) | 7.7 days | 1.4 hours |
| Annual GPU occupation (R=2) | 384 hours | 2.9 hours |

All benchmarks on NVIDIA L40 GPU, 2× Intel Xeon Gold 5420+ (56 cores), 502 GB RAM,
Oracle Linux 9.7 (containerised) on Rocky Linux 10.1 host, CUDA 13.1.

## Repository structure

```
iws-pipeline-comparison-gmd/
├── README.md
├── LICENSE
├── .gitignore
│
├── figures/
│   ├── figure-1_system_arch_v2.png      IWS architecture diagram
│   ├── figure-1_system_arch_v2.svg      IWS architecture diagram (vector)
│   ├── figure-2_map_v1.png             Global ISW detection map
│   ├── figure-3_roc_curves.pdf         ROC curves (generated)
│   ├── figure-3_roc_curves.png         ROC curves (generated)
│   └── plot_roc_curves.jl              Script to regenerate Figure 3
│
├── data/
│   ├── README.md                        File-to-paper mapping
│   ├── pipeline_a_gpu_evaluation.json
│   ├── pipeline_a_gpu_inference.json
│   ├── pipeline_a_gpu_power.json
│   ├── pipeline_a_cpu_evaluation.json
│   ├── pipeline_a_cpu_inference.json
│   ├── pipeline_b_gpu_evaluation.json
│   ├── pipeline_b_gpu_inference.json
│   ├── pipeline_b_gpu_power.json
│   ├── pipeline_b_cpu_evaluation.json
│   └── pipeline_b_cpu_inference.json
│
├── models/
│   ├── pipeline_b_sar_cnn_v2.jld2       Julia CNN checkpoint (v2, 2 March 2026)
│   └── pipeline_a_xgboost.json          XGBoost classifier
│
└── benchmarking/
    ├── README.md                        Execution environment and protocol
    ├── pipeline_a_gpu.py                Python GPU: inference + evaluation + power
    ├── pipeline_a_cpu.py                Python CPU: inference + evaluation
    ├── pipeline_b_gpu_inference.jl      Julia GPU: inference + power profiling
    ├── pipeline_b_gpu_evaluate.jl       Julia GPU: evaluation
    ├── pipeline_b_gpu_power_profiler.jl Julia GPU: power measurement module
    ├── pipeline_b_cpu_inference.jl      Julia CPU: inference
    ├── pipeline_b_cpu_evaluate.jl       Julia CPU: evaluation
    ├── pipeline_b_model.jl              SAR_CNN v2 architecture (283,329 params)
    └── pipeline_b_config.jl             Shared constants (image size, batch size)
```

## Reproducing figures

### Figure 3: ROC curves

Requires Julia 1.10+ with CairoMakie and JSON3:

```bash
julia figures/plot_roc_curves.jl
```

Output: `figures/figure-3_roc_curves.pdf` and `.png`.

### Figure 1: IWS architecture

Static diagram. Source files in `figures/`.

### Figure 2: Global ISW detection map

Generated from the IWS validation platform. Source data not included in this
repository; see the paper for details.

## Test set

The evaluation dataset (5,860 images: 2,930 ISW positives, 2,930 negatives)
was held out from all training. The dataset is curated and published by the
IWS with a persistent DOI (see paper for citation).

## Citation

If you use these data or scripts, please cite:

```
Pinelo, J., Shukla, A., Titericz, G., Santos-Ferreira, A., Gonçalves, J.,
and Moniz, J.: Choosing an operational inference pipeline for internal solitary
wave detection in Sentinel-1 SAR imagery: EVA02+XGBoost versus a Julia Lux CNN,
Geosci. Model Dev., submitted, 2026.
```

## Script authorship

Benchmarking scripts were developed by Arun Shukla (Pipeline B Julia scripts,
power profiling) and Gilberto Titericz (Pipeline A Python pipeline), under the
supervision of João Pinelo. Scripts were revised by João Pinelo for quality
assurance, metadata corrections, and repository structuring. Individual script
headers document specific contributions.

## Licence

MIT. See [LICENSE](LICENSE).

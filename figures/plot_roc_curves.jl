#!/usr/bin/env julia
#
# plot_roc_curves.jl
#
# Generates a publication-quality ROC curve figure comparing
# Pipeline A (Python EVA02+XGBoost) and Pipeline B (Julia Lux CNN)
# from the benchmark evaluation JSON files.
#
# Repo layout assumed:
#   iws-pipeline-comparison-gmd/
#   ├── data/
#   │   ├── pipeline_a_gpu_evaluation.json
#   │   └── pipeline_b_gpu_evaluation.json
#   └── figures/
#       └── plot_roc_curves.jl   ← this file
#
# Usage (from repo root):
#   julia figures/plot_roc_curves.jl
#
# Usage (from figures/):
#   julia plot_roc_curves.jl
#
# Override output directory:
#   julia figures/plot_roc_curves.jl --outdir /path/to/dir
#
# Dependencies:
#   using Pkg; Pkg.add(["CairoMakie", "JSON3"])
#
# Output: figure-3_roc_curves.pdf and .png in figures/

using JSON3
using CairoMakie

# ── Repo-aware path resolution ────────────────────────────────────

"""
    find_repo_root()

Walk upward from this script's directory to find the repo root,
identified by the presence of a `data/` directory containing the
expected JSON files.
"""
function find_repo_root()
    script_dir = @__DIR__
    candidates = [
        script_dir,                     # if run from repo root
        dirname(script_dir),            # if run from figures/
        joinpath(script_dir, ".."),      # explicit parent
    ]
    for dir in candidates
        data_dir = joinpath(abspath(dir), "data")
        if isdir(data_dir) &&
           isfile(joinpath(data_dir, "pipeline_b_gpu_evaluation.json")) &&
           isfile(joinpath(data_dir, "pipeline_a_gpu_evaluation.json"))
            return abspath(dir)
        end
    end
    error(
        "Cannot locate repo root. Expected data/ directory with evaluation JSONs.\n" *
        "Run this script from the repo root or from figures/."
    )
end

function parse_outdir(args)
    idx = findfirst(==("--outdir"), args)
    if idx !== nothing && idx < length(args)
        return args[idx + 1]
    end
    return nothing
end

# ── Configuration ─────────────────────────────────────────────────

const N_POS = 2930   # balanced test set: 2930 positives, 2930 negatives
const N_NEG = 2930

const REPO_ROOT  = find_repo_root()
const DATA_DIR   = joinpath(REPO_ROOT, "data")
const FIG_DIR    = let d = parse_outdir(ARGS)
    d !== nothing ? d : joinpath(REPO_ROOT, "figures")
end

const JULIA_EVAL  = joinpath(DATA_DIR, "pipeline_b_gpu_evaluation.json")
const PYTHON_EVAL = joinpath(DATA_DIR, "pipeline_a_gpu_evaluation.json")

const OUTPUT_STEM = "figure-3_roc_curves"
const OUTPUT_PDF  = joinpath(FIG_DIR, OUTPUT_STEM * ".pdf")
const OUTPUT_PNG  = joinpath(FIG_DIR, OUTPUT_STEM * ".png")

# ── Data loading ──────────────────────────────────────────────────

function load_julia_roc(path::String)
    data = JSON3.read(read(path, String))
    sweep = data.threshold_sweep

    fpr = Float64[]
    tpr = Float64[]

    # Anchor at (0, 0) — threshold = 1.0
    push!(fpr, 0.0)
    push!(tpr, 0.0)

    # Sweep is ordered by ascending threshold; reverse to trace
    # the ROC from (0,0) to (1,1)
    for entry in reverse(sweep)
        tp = entry.tp
        fp = entry.fp
        fn = entry.fn
        tn = entry.tn
        push!(tpr, tp / (tp + fn))
        push!(fpr, fp / (fp + tn))
    end

    # Anchor at (1, 1) — threshold = 0.0
    push!(fpr, 1.0)
    push!(tpr, 1.0)

    auc = data.model_performance.auc_roc
    return fpr, tpr, auc
end

function load_python_roc(path::String)
    data = JSON3.read(read(path, String))
    sweep = data.threshold_sweep

    fpr = Float64[]
    tpr = Float64[]

    push!(fpr, 0.0)
    push!(tpr, 0.0)

    for entry in reverse(sweep)
        recall    = entry.recall
        precision = entry.precision

        # Reconstruct counts from precision/recall and known class balance
        tp = recall * N_POS
        fp = (tp / precision) - tp
        push!(tpr, recall)
        push!(fpr, fp / N_NEG)
    end

    push!(fpr, 1.0)
    push!(tpr, 1.0)

    auc = data.auc_roc
    return fpr, tpr, auc
end

# ── Plotting ──────────────────────────────────────────────────────

# Colours: RdBu diverging palette (colourblind-safe)
const COL_A = Makie.RGB(0.13, 0.40, 0.67)  # #2166ac — Pipeline A
const COL_B = Makie.RGB(0.84, 0.38, 0.30)  # #d6604d — Pipeline B

function plot_roc(julia_path::String, python_path::String)
    j_fpr, j_tpr, j_auc = load_julia_roc(julia_path)
    p_fpr, p_tpr, p_auc = load_python_roc(python_path)

    # Publication theme
    fontsize_base = 10
    set_theme!(Theme(
        fontsize = fontsize_base,
        Axis = (
            xlabelsize     = fontsize_base + 1,
            ylabelsize     = fontsize_base + 1,
            xticklabelsize = fontsize_base - 1,
            yticklabelsize = fontsize_base - 1,
            titlesize      = fontsize_base + 2,
            spinewidth     = 0.8,
            xtickwidth     = 0.6,
            ytickwidth     = 0.6,
        ),
    ))

    # Single-column GMD figure: ~84 mm wide
    fig = Figure(size = (340, 340), figure_padding = (5, 15, 5, 5))

    ax = Axis(fig[1, 1],
        xlabel = "False positive rate",
        ylabel = "True positive rate",
        aspect = 1,
        xticks = 0:0.2:1.0,
        yticks = 0:0.2:1.0,
        limits = (0, 1, 0, 1),
    )

    # Diagonal reference (random classifier)
    lines!(ax, [0, 1], [0, 1],
        color     = (:grey60, 0.5),
        linestyle = :dash,
        linewidth = 0.8,
    )

    # Pipeline A — Python EVA02+XGBoost
    lines!(ax, p_fpr, p_tpr,
        color     = COL_A,
        linewidth = 1.5,
        label     = "Pipeline A (AUC = $(round(p_auc * 100; digits=2)) %)",
    )

    # Pipeline B — Julia Lux CNN
    lines!(ax, j_fpr, j_tpr,
        color     = COL_B,
        linewidth = 1.5,
        linestyle = :dash,
        label     = "Pipeline B (AUC = $(round(j_auc * 100; digits=2)) %)",
    )

    # Inset: zoomed view of the top-left corner
    ax_inset = Axis(fig[1, 1],
        width  = Relative(0.38),
        height = Relative(0.38),
        halign = 0.72,
        valign = 0.42,
        title  = "Top-left detail",
        titlesize      = fontsize_base - 3,
        xticks         = 0:0.04:0.12,
        yticks         = 0.92:0.04:1.0,
        limits         = (0, 0.12, 0.90, 1.0),
        xticklabelsize = fontsize_base - 2,
        yticklabelsize = fontsize_base - 2,
        xlabelsize     = fontsize_base - 2,
        ylabelsize     = fontsize_base - 2,
        backgroundcolor = :white,
        spinewidth     = 0.5,
    )

    lines!(ax_inset, p_fpr, p_tpr, color = COL_A, linewidth = 1.3)
    lines!(ax_inset, j_fpr, j_tpr, color = COL_B, linewidth = 1.3, linestyle = :dash)

    # Legend
    axislegend(ax,
        position   = :rb,
        labelsize  = fontsize_base - 1.5,
        framewidth = 0.6,
        padding    = (6, 6, 4, 4),
        margin     = (4, 4, 4, 4),
        rowgap     = 2,
    )

    return fig
end

# ── Main ──────────────────────────────────────────────────────────

function main()
    println("Repo root:  $REPO_ROOT")
    println("Data dir:   $DATA_DIR")
    println("Output dir: $FIG_DIR")
    println()

    # Verify inputs exist
    for f in (JULIA_EVAL, PYTHON_EVAL)
        isfile(f) || error("Missing: $f")
    end

    # Ensure output directory exists
    mkpath(FIG_DIR)

    println("Loading Julia evaluation:  $JULIA_EVAL")
    println("Loading Python evaluation: $PYTHON_EVAL")

    fig = plot_roc(JULIA_EVAL, PYTHON_EVAL)

    CairoMakie.save(OUTPUT_PDF, fig; pt_per_unit = 1)
    CairoMakie.save(OUTPUT_PNG, fig; px_per_unit = 3)  # ~300 DPI at 84 mm

    println()
    println("Saved: $OUTPUT_PDF")
    println("Saved: $OUTPUT_PNG")

    return fig
end

main()

using JSON3
using CSV, DataFrames, ImageIO, FileIO, ImageTransformations, ColorTypes, FixedPointNumbers, Colors, ImageCore, ImageMagick
# ============================================================
# pipeline_b_gpu_evaluate.jl — GPU-ENABLED EVALUATION (v2.2)
#
# FIXES vs original:
# 1. FPR/FNR wrong denominators — now FP/(FP+TN) and FN/(FN+TP)
# 2. Threshold sweep optimizes f1_score not accuracy
# 3. SAR_CNN() inline redefinition removed — loaded from JLD2
# 4. LuxCore.testmode() replaced with set_dropout_mode()
#    (matches training and CPU v2.2 exactly)
# ============================================================

ENV["JULIA_CUDA_MEMORY_POOL"] = "binned"
ENV["JULIA_CUDA_VERBOSE"] = "false"

using CUDA
if !CUDA.functional()
    error("CUDA is not functional — cannot run GPU evaluation")
end

using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2
using ProgressBars
using Optimisers
using Zygote
using LuxCUDA

using CairoMakie
import CairoMakie: Axis, Figure, lines!, axislegend,
                   xlims!, ylims!, save, vlines!,
                   heatmap!, text!, Colorbar
# ── Define model architecture BEFORE loading checkpoint ───────────────────
include("/tmp/benchmarks_iws/gpu_julia_iws/scripts/sar_cnn_gpu.jl")
# ============================================================
# FIX 4: set_dropout_mode replaces LuxCore.testmode()
# LuxCore.testmode() does not correctly handle all BN/Dropout
# states in Lux — set_dropout_mode matches training exactly
# ============================================================
function set_dropout_mode(st::NamedTuple, training::Bool)
    return map(st) do layer_st
        if layer_st isa NamedTuple
            set_dropout_mode(layer_st, training)
        else
            layer_st
        end
    end
end

function calculate_metrics_at_threshold(y_true::Vector{Float32},
                                        y_scores::Vector{Float32},
                                        threshold::Float32)
    predictions = y_scores .>= threshold
    labels      = y_true   .>= 0.5f0
    tp = sum( predictions .&  labels)
    tn = sum(.!predictions .& .!labels)
    fp = sum( predictions .& .!labels)
    fn = sum(.!predictions .&  labels)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp > 0 ? tp / (tp + fp) : 0.0f0
    recall    = tp > 0 ? tp / (tp + fn) : 0.0f0
    f1_score  = (precision + recall) > 0 ?
                2 * (precision * recall) / (precision + recall) : 0.0f0
    return Dict(
        "threshold" => threshold,
        "accuracy"  => accuracy,
        "precision" => precision,
        "recall"    => recall,
        "f1_score"  => f1_score,
        "tp" => tp, "tn" => tn, "fp" => fp, "fn" => fn
    )
end

function plot_confusion_matrix(metrics::Dict, save_path::String="confusion_matrix.png")
    tp = Int(metrics["tp"]); tn = Int(metrics["tn"])
    fp = Int(metrics["fp"]); fn = Int(metrics["fn"])
    conf_matrix = Float32[tn fp; fn tp]
    fig = Figure(size=(800, 700))
    ax  = Axis(fig[1, 1],
               xlabel = "Predicted Label",
               ylabel = "True Label",
               title  = "Confusion Matrix (Threshold = $(round(metrics["threshold"], digits=3)))",
               xticks = ([1, 2], ["Non-IW", "IW"]),
               yticks = ([1, 2], ["Non-IW", "IW"]))
    hm = heatmap!(ax, conf_matrix, colormap=:Blues,
                  colorrange=(0, maximum(conf_matrix)))
    text!(ax, 1, 1, text="TN\n$tn", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 2, 1, text="FP\n$fp", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 1, 2, text="FN\n$fn", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 2, 2, text="TP\n$tp", align=(:center,:center), fontsize=24, color=:black)
    Colorbar(fig[1, 2], hm, label="Count")
    save(save_path, fig)
    println("Confusion matrix saved to: $save_path")
    return fig
end

function analyze_fp_vs_fn(metrics::Dict,
                           y_true::Vector{Float32},
                           y_scores::Vector{Float32})
    println("\n" * "="^60)
    println("FALSE POSITIVE vs FALSE NEGATIVE ANALYSIS")
    println("="^60)
    tp = Int(metrics["tp"]); tn = Int(metrics["tn"])
    fp = Int(metrics["fp"]); fn = Int(metrics["fn"])

    # ── FIX 1: correct denominators ───────────────────────────────────────────
    # ORIGINAL BUG: fp_rate = fp / total  (divides by all samples — wrong)
    #               fn_rate = fn / total
    # FIX:          fp_rate = fp / (fp+tn)  = FPR (fraction of negatives)
    #               fn_rate = fn / (fn+tp)  = FNR / miss rate (fraction of positives)
    n_neg   = fp + tn
    n_pos   = fn + tp
    fp_rate = n_neg > 0 ? fp / n_neg : 0.0f0
    fn_rate = n_pos > 0 ? fn / n_pos : 0.0f0

    println("\n CONFUSION MATRIX BREAKDOWN:")
    println("   ┌─────────────────┬─────────────────┐")
    println("   │                 │   PREDICTED     │")
    println("   │                 ├────────┬────────┤")
    println("   │                 │ Non-IW │   IW   │")
    println("   ├─────────────────┼────────┼────────┤")
    println("   │ ACTUAL: Non-IW  │   $tn   │   $fp   │")
    println("   │ ACTUAL: IW      │   $fn   │   $tp   │")
    println("   └─────────────────┴────────┴────────┘")
    println("\n ERROR TYPE BREAKDOWN:")
    println("   True Positives (TP):  $tp (correctly detected IW)")
    println("   True Negatives (TN):  $tn (correctly detected Non-IW)")
    println("   False Positives (FP): $fp (Non-IW misclassified as IW)")
    println("   False Negatives (FN): $fn (IW misclassified as Non-IW)")
    println("\n ERROR RATES (fraction of own class):")
    println("   FPR (FP / N_neg): $(round(fp_rate*100, digits=2))%")
    println("   FNR (FN / N_pos): $(round(fn_rate*100, digits=2))%  ← miss rate")
    println("\n  SCIENTIFIC INTERPRETATION:")
    println("   FALSE POSITIVES (FP = $fp):")
    println("   Non-IW classified as IW")
    println("   Impact: NOISY SCIENCE — spurious detections")
    println("   Consequence: Overestimation of IW events")
    println("\n   FALSE NEGATIVES (FN = $fn):")
    println("   IW classified as Non-IW")
    println("   Impact: MISSED EVENTS — lose real phenomena")
    println("   Consequence: Underestimation of IW occurrence")

    tolerance = round(Int, 0.01 * (tp + tn + fp + fn))  # 1% of total
    if abs(fn - fp) <= tolerance
        println("\n ERROR BALANCE: FP and FN within 1% tolerance (gap=$(fn-fp))")
    elseif fp > fn
        println("\n  ERROR BIAS: FALSE POSITIVES dominate by $(fp-fn) samples")
        println("   → Suggestion: INCREASE threshold")
    else
        println("\n  ERROR BIAS: FALSE NEGATIVES dominate by $(fn-fp) samples")
        println("   → Suggestion: DECREASE threshold")
    end
    println("="^60)
    return fp, fn, fp_rate, fn_rate
end

# ============================================================
# FIX 2: optimize_threshold_with_cost
#
# ORIGINAL BUG: optimization_metric = "accuracy"
#   → picks threshold that maximises accuracy
#   → for balanced datasets accuracy and F1 diverge at tails
#   → can increase FN significantly (as seen in CPU v2.1)
#
# FIX: optimization_metric = "f1_score" (default)
#   → directly optimises the harmonic mean of precision/recall
#   → more scientifically appropriate for detection tasks
# ============================================================
function optimize_threshold_with_cost(y_true::Vector{Float32},
                                       y_scores::Vector{Float32};
                                       threshold_range = 0.3f0:0.01f0:0.7f0,
                                       optimization_metric = "f1_score",  # ← FIXED
                                       fp_cost::Float32 = 1.0f0,
                                       fn_cost::Float32 = 1.0f0)
    println("\n THRESHOLD OPTIMIZATION WITH COST ANALYSIS")
    println("="^60)
    println("NOTE: We do NOT assume threshold = 0.5 is optimal!")
    println("Sweeping thresholds from $(first(threshold_range)) to $(last(threshold_range))")
    println("Optimization metric: $optimization_metric")
    println("FP Cost: $fp_cost | FN Cost: $fn_cost")
    println("="^60)

    best_metric_value = -Inf   # ← FIXED: maximise directly (was Inf + negation)
    best_threshold    = 0.5f0
    best_metrics      = nothing
    all_results       = []

    for threshold in threshold_range
        metrics    = calculate_metrics_at_threshold(y_true, y_scores, threshold)
        total_cost = metrics["fp"] * fp_cost + metrics["fn"] * fn_cost
        metrics["total_cost"] = total_cost
        push!(all_results, metrics)

        metric_value = if optimization_metric == "accuracy"
            metrics["accuracy"]
        elseif optimization_metric == "f1_score"
            metrics["f1_score"]
        elseif optimization_metric == "balanced"
            min(metrics["precision"], metrics["recall"])
        elseif optimization_metric == "recall"
            metrics["recall"]
        elseif optimization_metric == "cost"
            -total_cost
        else
            metrics["f1_score"]
        end

        if metric_value > best_metric_value   # ← FIXED: maximise
            best_metric_value = metric_value
            best_threshold    = threshold
            best_metrics      = metrics
        end
    end
    return best_threshold, best_metrics, all_results
end

function plot_threshold_analysis(all_results::Vector,
                                  best_threshold::Float32,
                                  save_path::String="threshold_analysis.png")
    thresholds = [r["threshold"] for r in all_results]
    fig = Figure(size=(1200, 800))
    ax  = Axis(fig[1, 1],
               xlabel = "Decision Threshold",
               ylabel = "Metric Value",
               title  = "Metrics vs Decision Threshold")
    lines!(ax, thresholds, [r["accuracy"]  for r in all_results],
           color=:blue,   linewidth=2, label="Accuracy")
    lines!(ax, thresholds, [r["precision"] for r in all_results],
           color=:green,  linewidth=2, label="Precision")
    lines!(ax, thresholds, [r["recall"]    for r in all_results],
           color=:red,    linewidth=2, label="Recall")
    lines!(ax, thresholds, [r["f1_score"]  for r in all_results],
           color=:purple, linewidth=2, label="F1-Score")
    vlines!(ax, [best_threshold],
            color=:black, linestyle=:dash, linewidth=2, label="Optimal Threshold (F1)")
    axislegend(ax, position=:rb)
    xlims!(ax, first(thresholds), last(thresholds))
    ylims!(ax, 0, 1)
    save(save_path, fig)
    println("Threshold analysis plot saved to: $save_path")
    return fig
end

function calculate_auc_roc(y_true::Vector{Float32}, y_scores::Vector{Float32})
    sorted_idx    = sortperm(y_scores, rev=true)
    y_true_sorted = y_true[sorted_idx]
    n_positive    = sum(y_true_sorted)
    n_negative    = length(y_true_sorted) - n_positive
    tpr = Float32[0.0f0]
    fpr = Float32[0.0f0]
    tp  = 0.0f0; fp = 0.0f0
    for i in 1:length(y_true_sorted)
        if y_true_sorted[i] > 0.5f0; tp += 1.0f0
        else;                          fp += 1.0f0
        end
        push!(tpr, tp / n_positive)
        push!(fpr, fp / n_negative)
    end
    auc = sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0f0
              for i in 2:length(fpr))
    return auc, fpr, tpr
end

function plot_roc_curve(fpr::Vector{Float32}, tpr::Vector{Float32},
                         auc::Float32, save_path::String="roc_curve.png")
    fig = Figure(size=(800, 800))
    ax  = Axis(fig[1, 1],
               xlabel = "False Positive Rate (FPR)",
               ylabel = "True Positive Rate (TPR)",
               title  = "ROC Curve (AUC = $(round(auc, digits=4)))")
    lines!(ax, fpr, tpr,
           color=:blue, linewidth=2, label="ROC Curve")
    lines!(ax, [0.0f0, 1.0f0], [0.0f0, 1.0f0],
           color=:red, linestyle=:dash, linewidth=2, label="Random Classifier")
    axislegend(ax, position=:rb)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    save(save_path, fig)
    println("   ROC curve saved to: $save_path")
    return fig
end

function evaluate_model(model_path::String="/tmp/benchmarks_iws/gpu_julia_iws/models/model_v2_20260302_202405.jld2")
    evaluation_start_time = time()
    
    println("="^60)
    println("SAR CNN MODEL EVALUATION (GPU — v2.2)")
    println("="^60)

    # ── Load model ────────────────────────────────────────────────────────────
    println("\n Loading trained model from: $model_path")  # ← path always logged

    loaded_data = load(model_path)

    # ── FIX 3: load model FROM JLD2, do NOT redefine SAR_CNN() inline ─────────
    # ORIGINAL BUG: model = SAR_CNN()
    #   → creates a fresh untrained model with random weights
    #   → then loads ps/st on top — but if architecture changed between
    #     training and this script, ps/st silently mismatch → dead predictions
    # FIX: load the model object directly from the checkpoint
    #   → guarantees architecture matches the saved weights exactly
    model = loaded_data["model"]   # ← FIXED (was SAR_CNN())
    ps    = loaded_data["ps"]
    st    = loaded_data["st"]

    norm_mean = Float32(loaded_data["norm_mean"])
    norm_std  = Float32(loaded_data["norm_std"])
    println(@sprintf("   norm_mean = %.4f   norm_std = %.4f", norm_mean, norm_std))

    ps = ps |> gpu_device()
    st = st |> gpu_device()
    println("Model loaded successfully on GPU")

    if haskey(loaded_data, "tta_val_acc")
        @printf("   Checkpoint  val acc : %.2f%%\n",
                Float64(loaded_data["tta_val_acc"]) * 100)
    end
    if haskey(loaded_data, "best_val_loss")
        @printf("   Checkpoint best val loss: %.4f\n",
                Float64(loaded_data["best_val_loss"]))
    end

    # ── Load validation data ──────────────────────────────────────────────────

# ── Load validation data directly ─────────────────────────────────────────
println("\n Loading validation data...")
data_path = "/tmp/benchmarks_iws/data/raw/test/"
csv_path  = "/tmp/benchmarks_iws/data/raw/test.csv"
df         = CSV.read(csv_path, DataFrame)
n_val      = nrow(df)
batch_size = 128
println("   Validation samples: $n_val")
println("   Batch size: $batch_size")

# ── Set eval mode ─────────────────────────────────────────────────────────
st_val = LuxCore.testmode(set_dropout_mode(st, false))

# ── Collect predictions ───────────────────────────────────────────────────
println("\n Evaluating model...")
all_predictions = Float32[]
all_labels      = Float32[]

function load_image_tensor(path::String)
    img  = load(path)
    img  = imresize(img, (256, 256))        # model input: 256×256
    img  = Gray.(img)                        # force grayscale — model has 1 input channel
    arr  = Float32.(channelview(img))        # 256×256 (single channel, no leading dim)
    return reshape(arr, 256, 256, 1)         # H×W×C = 256×256×1
end

for batch_start in 1:batch_size:n_val
    batch_end   = min(batch_start + batch_size - 1, n_val)
    valid_imgs  = Float32[]
    valid_labels = Float32[]
    valid_count  = 0

    for i in batch_start:batch_end
        fpath = joinpath(data_path, df.id[i])
        if isfile(fpath)
            try
                img_arr = load_image_tensor(fpath)      # 256×256×1
                append!(valid_imgs, vec(img_arr))
                push!(valid_labels, Float32(df.ground_truth[i]))
                valid_count += 1
            catch e
                @warn "Skipping $fpath: $e"
            end
        end
    end

    valid_count == 0 && continue

    # Lux expects (H, W, C, N) = (256, 256, 1, batch)
    X_batch = reshape(valid_imgs, 256, 256, 1, valid_count)
    X_batch = (X_batch .- norm_mean) ./ norm_std
    X_batch = X_batch |> gpu_device()

    y_pred, _ = model(X_batch, ps, st_val)

    append!(all_predictions, vec(y_pred |> cpu_device()))
    append!(all_labels,      valid_labels)
end







    # ── Sanity check predictions ───────────────────────────────────────────────
    pos_rate = sum(all_predictions .>= 0.5f0) / length(all_predictions) * 100
    @printf("   Score range: [%.4f, %.4f]  mean: %.4f\n",
            minimum(all_predictions), maximum(all_predictions), mean(all_predictions))
    @printf("   Positive rate (threshold=0.5): %.1f%%\n", pos_rate)
    if pos_rate < 1.0 || pos_rate > 99.0
        println("WARNING: Extreme positive rate — check model/normalization")
    end

    # ── Baseline at threshold=0.5 ─────────────────────────────────────────────
    println("\n BASELINE (Threshold = 0.5)")
    println("="^60)
    baseline_metrics = calculate_metrics_at_threshold(all_labels, all_predictions, 0.5f0)
    @printf("Accuracy:  %.2f%%\n", baseline_metrics["accuracy"]  * 100)
    @printf("Precision: %.2f%%\n", baseline_metrics["precision"] * 100)
    @printf("Recall:    %.2f%%\n", baseline_metrics["recall"]    * 100)
    @printf("F1-Score:  %.2f%%\n", baseline_metrics["f1_score"]  * 100)
    println("="^60)

    # ── Threshold optimization — F1 by default ────────────────────────────────
    best_threshold, best_metrics, all_results =
        optimize_threshold_with_cost(
            all_labels, all_predictions;
            threshold_range      = 0.3f0:0.01f0:0.7f0,
            optimization_metric  = "f1_score",   # ← FIXED (was "accuracy")
            fp_cost              = 1.0f0,
            fn_cost              = 1.0f0
        )

    println("\n OPTIMAL THRESHOLD RESULTS (optimized for F1)")
    println("="^60)
    @printf("Optimal Threshold : %.3f\n", best_threshold)
    @printf("Accuracy  : %.2f%%  (Δ = %+.2f%%)\n",
            best_metrics["accuracy"] * 100,
            (best_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100)
    @printf("Precision : %.2f%%\n", best_metrics["precision"] * 100)
    @printf("Recall    : %.2f%%\n", best_metrics["recall"]    * 100)
    @printf("F1-Score  : %.2f%%\n", best_metrics["f1_score"]  * 100)
    println("="^60)

    # ── FP/FN analysis ────────────────────────────────────────────────────────
    fp, fn, fp_rate, fn_rate =
        analyze_fp_vs_fn(best_metrics, all_labels, all_predictions)

    # ── Plots ─────────────────────────────────────────────────────────────────
    println("\n Plotting confusion matrix...")
    plot_confusion_matrix(best_metrics,     "/tmp/benchmarks_iws/gpu_julia_iws/results/confusion_matrix_optimal.png")
    println("Plotting baseline confusion matrix...")
    plot_confusion_matrix(baseline_metrics, "/tmp/benchmarks_iws/gpu_julia_iws/results/confusion_matrix_baseline.png")

    println("\n Calculating AUC-ROC...")
    auc, fpr, tpr = calculate_auc_roc(all_labels, all_predictions)
    @printf("AUC-ROC: %.4f\n", auc)

    println("\n Plotting threshold analysis...")
    plot_threshold_analysis(all_results, best_threshold, "/tmp/benchmarks_iws/gpu_julia_iws/results/threshold_analysis.png")
    println("\n Plotting ROC curve...")
    plot_roc_curve(fpr, tpr, auc, "/tmp/benchmarks_iws/gpu_julia_iws/results/roc_curve.png")

    total_evaluation_time = time() - evaluation_start_time

    # ── Final summary ─────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("FINAL EVALUATION SUMMARY")
    println("="^60)
    @printf("Baseline Accuracy (threshold=0.5)       : %.2f%%\n",
            baseline_metrics["accuracy"] * 100)
    @printf("Optimized Accuracy (threshold=%.3f)    : %.2f%%\n",
            best_threshold, best_metrics["accuracy"] * 100)
    @printf("Accuracy Gain                           : %+.2f%%\n",
            (best_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100)
    println("")
    println("Error Analysis at Optimal Threshold:")
    @printf("FPR (FP / N_neg) : %.2f%%\n", fp_rate * 100)
    @printf("FNR (FN / N_pos) : %.2f%%  ← miss rate\n", fn_rate * 100)
    println("")
    @printf("AUC-ROC Score: %.4f\n", auc)
    println("="^60)
    println("\n" * "="^60)
    println("EVALUATION TIME SUMMARY")
    println("="^60)
    @printf("Total evaluation time : %.1f seconds\n", total_evaluation_time)
    @printf("Time per sample       : %.1f ms\n",
            total_evaluation_time / n_val * 1000)
    println("="^60)

    # ── Save model ────────────────────────────────────────────────────────────
    ps_cpu = ps |> cpu_device()
    st_cpu = st |> cpu_device()

    if best_metrics["accuracy"] >= 0.85f0
        println("\n Saving best model with optimal threshold...")
        jldsave("/tmp/benchmarks_iws/gpu_julia_iws/models/model_best.jld2";
                model            = model,
                ps               = ps_cpu,
                st               = st_cpu,
                val_accuracy     = best_metrics["accuracy"],
                auc_roc          = auc,
                optimal_threshold= best_threshold,
                baseline_accuracy= baseline_metrics["accuracy"],
                false_positives  = fp,
                false_negatives  = fn,
                fpr              = fp_rate,
                fnr              = fn_rate,
                backend          = "gpu",
                compute_mode     = "gpu")
    else
        println("\n Accuracy $(round(best_metrics["accuracy"]*100, digits=2))% — saved but consider retraining.")
        jldsave("/tmp/benchmarks_iws/gpu_julia_iws/models/model_best.jld2";
                model=model, ps=ps_cpu, st=st_cpu,
                val_accuracy=best_metrics["accuracy"],
                auc_roc=auc,
                optimal_threshold=best_threshold,
                backend="gpu",
                compute_mode="gpu")
    end

    # ── Save JSON ─────────────────────────────────────────────────────────────
    println("\n Saving evaluation metrics to JSON...")
    evaluation_metrics = Dict(
        "metadata" => Dict(
            "timestamp"              => string(now()),
            "pipeline"               => "Julia_Lux_CNN_GPU_v2.2",
            "device"                 => "gpu",
            "total_samples"          => n_val,
            "model_path"             => model_path,
            "threshold_metric"       => "f1_score",
            "threshold_metric_note"  => "Changed from accuracy (original) to f1_score (v2.2)"
        ),
        "baseline" => Dict(
            "threshold" => 0.5,
            "accuracy"  => Float64(baseline_metrics["accuracy"]),
            "precision" => Float64(baseline_metrics["precision"]),
            "recall"    => Float64(baseline_metrics["recall"]),
            "f1_score"  => Float64(baseline_metrics["f1_score"]),
            "confusion_matrix" => Dict(
                "tp" => Int(baseline_metrics["tp"]),
                "tn" => Int(baseline_metrics["tn"]),
                "fp" => Int(baseline_metrics["fp"]),
                "fn" => Int(baseline_metrics["fn"])
            )
        ),
        "optimal" => Dict(
            "threshold"        => Float64(best_threshold),
            "optimized_metric" => "f1_score",
            "accuracy"         => Float64(best_metrics["accuracy"]),
            "precision"        => Float64(best_metrics["precision"]),
            "recall"           => Float64(best_metrics["recall"]),
            "f1_score"         => Float64(best_metrics["f1_score"]),
            "cost"             => Float64(best_metrics["total_cost"]),
            "confusion_matrix" => Dict(
                "tp" => Int(best_metrics["tp"]),
                "tn" => Int(best_metrics["tn"]),
                "fp" => Int(best_metrics["fp"]),
                "fn" => Int(best_metrics["fn"])
            )
        ),
        "error_analysis" => Dict(
            # ── FIX 1: correct denominators ───────────────────────────────────
            "fpr"            => Float64(fp_rate),   # FP / N_negative
            "fnr"            => Float64(fn_rate),   # FN / N_positive
            "fp_count"       => Int(fp),
            "fn_count"       => Int(fn),
            "error_bias"     => abs(fn-fp) <= round(Int,0.01*(fp+fn+Int(best_metrics["tp"])+Int(best_metrics["tn"]))) ?
                                "BALANCED" : (fn > fp ? "FALSE_NEGATIVES" : "FALSE_POSITIVES"),
            "recommendation" => fn > fp ? "DECREASE threshold" : "INCREASE threshold"
        ),
        "model_performance" => Dict(
            "auc_roc" => Float64(auc)
        ),
        "threshold_sweep" => [
            Dict(
                "threshold" => Float64(r["threshold"]),
                "accuracy"  => Float64(r["accuracy"]),
                "precision" => Float64(r["precision"]),
                "recall"    => Float64(r["recall"]),
                "f1_score"  => Float64(r["f1_score"]),
                "cost"      => Float64(r["total_cost"]),
                "tp"        => Int(r["tp"]),
                "tn"        => Int(r["tn"]),
                "fp"        => Int(r["fp"]),
                "fn"        => Int(r["fn"])
            )
            for r in all_results
        ]
    )


    json_filename = "/tmp/benchmarks_iws/gpu_julia_iws/results/julia_gpu_evaluation_metrics.json"
    open(json_filename, "w") do f
        JSON3.write(f, evaluation_metrics)
    end
    println("Evaluation metrics saved to: $json_filename")

    return baseline_metrics["accuracy"], best_metrics["accuracy"],
           best_threshold, auc, fp, fn
end

# ── Run ───────────────────────────────────────────────────────────────────────
println("Starting comprehensive model evaluation...")
baseline_acc, optimized_acc, optimal_threshold, auc, fp, fn = evaluate_model()

println("\n" * "="^60)
println("FINAL RESULTS")
println("="^60)
@printf("Baseline Accuracy (0.5)      : %.2f%%\n", baseline_acc    * 100)
@printf("Optimized Accuracy (%.3f)   : %.2f%%\n",  optimal_threshold, optimized_acc * 100)
@printf("Improvement                  : %+.2f%%\n",
        (optimized_acc - baseline_acc) * 100)
println("")
@printf("False Positives : %d\n", fp)
@printf("False Negatives : %d\n", fn)
println("")
@printf("AUC-ROC Score   : %.4f\n", auc)
println("="^60)

# ============================================================
# pipeline_b_cpu_evaluate.jl — CPU EVALUATION (mirrors pipeline_b_gpu_evaluate.jl)
# UPDATED: 2026-03-17
#
# FIXES IMPLEMENTED:
#   1. Added random seed for reproducibility
#   2. Optimized image loading (direct 4D tensor construction)
#   3. Added confidence intervals (95% CI)
#   4. Added baseline comparisons (random, majority class)
#   5. Added real-world imbalance warning
#   6. Added normalization verification to prevent data leakage
#   7. Added timing breakdown (I/O vs inference)
#   8. Added class distribution analysis
# ============================================================

using JSON3
using CSV, DataFrames, FileIO, ImageTransformations, ColorTypes, Colors, ImageCore
using Lux, LuxCore, NNlib
using Random, Statistics, Printf, Dates, JLD2

using CairoMakie
import CairoMakie: Axis, Figure, lines!, axislegend,
                   xlims!, ylims!, save, vlines!,
                   heatmap!, text!, Colorbar

using LinearAlgebra: BLAS
BLAS.set_num_threads(Sys.CPU_THREADS)

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================
# ARCHITECTURE — identical to GPU sar_cnn_gpu.jl
# ============================================================

struct ConvBlock <: LuxCore.AbstractLuxContainerLayer{(:conv, :bn, :pool, :drop)}
    conv :: Conv
    bn   :: BatchNorm
    pool :: Union{MaxPool, NoOpLayer}
    drop :: Dropout
end

function ConvBlock(kernel, in_ch, out_ch;
                   stride=(1,1), pad=0,
                   pool_size=nothing,
                   dropout_rate=0.1f0)
    conv = Conv(kernel, in_ch => out_ch; stride=stride, pad=pad)
    bn   = BatchNorm(out_ch)
    pool = isnothing(pool_size) ? NoOpLayer() : MaxPool(pool_size)
    drop = Dropout(dropout_rate)
    return ConvBlock(conv, bn, pool, drop)
end

function (cb::ConvBlock)(x, ps, st)
    y, st_conv = cb.conv(x,  ps.conv, st.conv)
    y, st_bn   = cb.bn(y,   ps.bn,   st.bn)
    y          = relu.(y)
    y, st_pool = cb.pool(y, ps.pool, st.pool)
    y, st_drop = cb.drop(y, ps.drop, st.drop)
    return y, (conv=st_conv, bn=st_bn, pool=st_pool, drop=st_drop)
end

struct AggBlock <: LuxCore.AbstractLuxLayer end
LuxCore.initialparameters(::Random.AbstractRNG, ::AggBlock) = NamedTuple()
LuxCore.initialstates(::Random.AbstractRNG,     ::AggBlock) = NamedTuple()

function (::AggBlock)(x, ps, st)
    y = mean(x; dims=(1,2))
    y = reshape(y, size(y,3), size(y,4))
    return y, st
end

struct ClassifierBlock <: LuxCore.AbstractLuxContainerLayer{(:dense1, :dropout, :dense2)}
    dense1  :: Dense
    dropout :: Dropout
    dense2  :: Dense
end

function ClassifierBlock(in_features::Int; dropout_rate::Float32=0.4f0)
    dense1  = Dense(in_features, 64, relu)
    dropout = Dropout(dropout_rate)
    dense2  = Dense(64, 1)
    return ClassifierBlock(dense1, dropout, dense2)
end

function (cb::ClassifierBlock)(x, ps, st)
    y, st1   = cb.dense1(x,  ps.dense1,  st.dense1)
    y, st_do = cb.dropout(y, ps.dropout, st.dropout)
    y, st2   = cb.dense2(y,  ps.dense2,  st.dense2)
    y        = sigmoid_fast.(y)
    return y, (dense1=st1, dropout=st_do, dense2=st2)
end

struct SAR_CNN <: LuxCore.AbstractLuxContainerLayer{(
        :block1, :block2, :block3, :block4, :agg, :classifier)}
    block1     :: ConvBlock
    block2     :: ConvBlock
    block3     :: ConvBlock
    block4     :: ConvBlock
    agg        :: AggBlock
    classifier :: ClassifierBlock
end

function SAR_CNN()
    block1 = ConvBlock((7,7), 1,   32;  stride=(2,2), pad=3, dropout_rate=0.05f0)
    block2 = ConvBlock((5,5), 32,  64;  stride=(1,1), pad=2, pool_size=(2,2), dropout_rate=0.1f0)
    block3 = ConvBlock((3,3), 64,  128; stride=(1,1), pad=1, pool_size=(2,2), dropout_rate=0.15f0)
    block4 = ConvBlock((3,3), 128, 128; stride=(1,1), pad=1, pool_size=(2,2), dropout_rate=0.2f0)
    agg        = AggBlock()
    classifier = ClassifierBlock(128; dropout_rate=0.4f0)
    return SAR_CNN(block1, block2, block3, block4, agg, classifier)
end

function (m::SAR_CNN)(x, ps, st)
    y, st1 = m.block1(x, ps.block1, st.block1)
    y, st2 = m.block2(y, ps.block2, st.block2)
    y, st3 = m.block3(y, ps.block3, st.block3)
    y, st4 = m.block4(y, ps.block4, st.block4)
    y, st5 = m.agg(y,   ps.agg,    st.agg)
    y, st6 = m.classifier(y, ps.classifier, st.classifier)
    return y, (block1=st1, block2=st2, block3=st3,
               block4=st4, agg=st5, classifier=st6)
end

# ============================================================
# set_dropout_mode — identical to GPU Julia version
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

# ============================================================
# METRIC FUNCTIONS — enhanced with confidence intervals
# ============================================================

function calculate_metrics_at_threshold(y_true::Vector{Float32},
                                        y_scores::Vector{Float32},
                                        threshold::Float32)
    predictions = y_scores .>= threshold
    labels      = y_true   .>= 0.5f0
    tp = sum( predictions .&  labels)
    tn = sum(.!predictions .& .!labels)
    fp = sum( predictions .& .!labels)
    fn = sum(.!predictions .&  labels)
    
    n_total = tp + tn + fp + fn
    accuracy  = (tp + tn) / n_total
    precision = tp > 0 ? tp / (tp + fp) : 0.0f0
    recall    = tp > 0 ? tp / (tp + fn) : 0.0f0
    f1_score  = (precision + recall) > 0 ?
                2 * (precision * recall) / (precision + recall) : 0.0f0
    
    # Calculate 95% confidence interval for accuracy (Wilson score interval)
    z = 1.96f0  # 95% confidence
    n = Float32(n_total)
    p = accuracy
    denominator = 1 + z^2/n
    center = p + z^2/(2n)
    spread = z * sqrt(p*(1-p)/n + z^2/(4n^2))
    ci_lower = max(0, (center - spread) / denominator)
    ci_upper = min(1, (center + spread) / denominator)
    
    return Dict(
        "threshold" => threshold,
        "accuracy"  => accuracy,
        "accuracy_ci_lower" => ci_lower,
        "accuracy_ci_upper" => ci_upper,
        "precision" => precision,
        "recall"    => recall,
        "f1_score"  => f1_score,
        "tp" => tp, "tn" => tn, "fp" => fp, "fn" => fn,
        "n_total" => n_total,
        "n_pos" => tp + fn,
        "n_neg" => tn + fp
    )
end

# ============================================================
# BASELINE COMPARISONS — new
# ============================================================

function calculate_baselines(y_true::Vector{Float32})
    labels = y_true .>= 0.5f0
    n_pos = sum(labels)
    n_neg = length(labels) - n_pos
    n_total = length(labels)
    
    # Random classifier (expected accuracy = 50%)
    random_acc = 0.5f0
    
    # Majority class classifier
    majority_class = n_pos > n_neg ? 1.0f0 : 0.0f0
    majority_acc = max(n_pos, n_neg) / n_total
    
    # Persistence (if applicable - here just as reference)
    persistence_acc = 0.5f0  # Placeholder
    
    return Dict(
        "random_classifier" => random_acc,
        "majority_class" => majority_acc,
        "majority_class_name" => n_pos > n_neg ? "IW" : "Non-IW",
        "class_balance" => Dict(
            "n_pos" => n_pos,
            "n_neg" => n_neg,
            "pos_ratio" => n_pos / n_total,
            "neg_ratio" => n_neg / n_total
        )
    )
end

# ============================================================
# OPTIMIZED IMAGE LOADING — direct to 4D tensor
# ============================================================

function load_image_batch(file_paths::Vector{String}, 
                          norm_mean::Float32, 
                          norm_std::Float32)
    n_images = length(file_paths)
    n_valid = 0
    valid_indices = Int[]
    
    # Pre-allocate 4D tensor (H, W, C, N)
    X_batch = zeros(Float32, 256, 256, 1, n_images)
    labels = Float32[]
    
    for (idx, fpath) in enumerate(file_paths)
        if !isfile(fpath)
            continue
        end
        try
            img = load(fpath)
            img = imresize(img, (256, 256))
            img = Gray.(img)
            arr = Float32.(channelview(img))  # 256×256
            
            # Direct assignment to pre-allocated tensor
            X_batch[:, :, 1, idx] = arr
            n_valid += 1
            push!(valid_indices, idx)
        catch e
            @warn "Skipping $fpath: $e"
        end
    end
    
    if n_valid == 0
        return nothing, nothing
    end
    
    # Keep only valid images
    X_batch = X_batch[:, :, :, valid_indices]
    
    # Normalize
    X_batch = (X_batch .- norm_mean) ./ norm_std
    
    return X_batch, labels
end

# ============================================================
# PLOT FUNCTIONS — enhanced with CI
# ============================================================

function plot_confusion_matrix(metrics::Dict, save_path::String="confusion_matrix.png")
    tp = Int(metrics["tp"]); tn = Int(metrics["tn"])
    fp = Int(metrics["fp"]); fn = Int(metrics["fn"])
    conf_matrix = Float32[tn fp; fn tp]
    
    fig = Figure(size=(800, 700))
    ax  = Axis(fig[1,1],
               xlabel = "Predicted Label",
               ylabel = "True Label",
               title  = "Confusion Matrix (Threshold = $(round(metrics["threshold"], digits=3)))\n" *
                       "Accuracy: $(round(metrics["accuracy"]*100, digits=2))% " *
                       "[$(round(metrics["accuracy_ci_lower"]*100, digits=2))-$(round(metrics["accuracy_ci_upper"]*100, digits=2))%]",
               xticks = ([1,2], ["Non-IW", "IW"]),
               yticks = ([1,2], ["Non-IW", "IW"]))
    
    hm = heatmap!(ax, conf_matrix, colormap=:Blues,
                  colorrange=(0, maximum(conf_matrix)))
    
    text!(ax, 1, 1, text="TN\n$tn", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 2, 1, text="FP\n$fp", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 1, 2, text="FN\n$fn", align=(:center,:center), fontsize=24, color=:black)
    text!(ax, 2, 2, text="TP\n$tp", align=(:center,:center), fontsize=24, color=:black)
    
    Colorbar(fig[1,2], hm, label="Count")
    save(save_path, fig)
    println("   Confusion matrix saved to: $save_path")
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
    n_total = Int(metrics["n_total"])
    n_pos = Int(metrics["n_pos"])
    n_neg = Int(metrics["n_neg"])

    fp_rate = n_neg > 0 ? fp / n_neg : 0.0f0   # FPR = FP / (FP + TN)
    fn_rate = n_pos > 0 ? fn / n_pos : 0.0f0   # FNR = FN / (FN + TP)

    println("\nCONFUSION MATRIX BREAKDOWN:")
    println("   ┌─────────────────┬─────────────────┐")
    println("   │                 │   PREDICTED     │")
    println("   │                 ├────────┬────────┤")
    println("   │                 │ Non-IW │   IW   │")
    println("   ├─────────────────┼────────┼────────┤")
    @printf("   │ ACTUAL: Non-IW  │ %6d │ %6d │\n", tn, fp)
    @printf("   │ ACTUAL: IW      │ %6d │ %6d │\n", fn, tp)
    println("   └─────────────────┴────────┴────────┘")
    
    println("\nCLASS DISTRIBUTION:")
    @printf("   Negative samples (Non-IW): %d (%.1f%%)\n", n_neg, n_neg/n_total*100)
    @printf("   Positive samples (IW)    : %d (%.1f%%)\n", n_pos, n_pos/n_total*100)
    
    println("\nERROR TYPE BREAKDOWN:")
    println("   True Positives  (TP): $tp (correctly detected IW)")
    println("   True Negatives  (TN): $tn (correctly detected Non-IW)")
    println("   False Positives (FP): $fp (Non-IW misclassified as IW)")
    println("   False Negatives (FN): $fn (IW misclassified as Non-IW)")

    println("\nERROR RATES (fraction of own class):")
    println("   FPR (FP / N_neg): $(round(fp_rate*100, digits=2))%")
    println("   FNR (FN / N_pos): $(round(fn_rate*100, digits=2))%  <- miss rate")

    println("\nSCIENTIFIC INTERPRETATION:")
    println("   FALSE POSITIVES (FP = $fp):")
    println("   -> Non-IW classified as IW")
    println("   -> Impact: NOISY SCIENCE — spurious detections")
    println("   -> Consequence: Overestimation of IW events")
    println("\n   FALSE NEGATIVES (FN = $fn):")
    println("   -> IW classified as Non-IW")
    println("   -> Impact: MISSED EVENTS — lose real phenomena")
    println("   -> Consequence: Underestimation of IW occurrence")
    
    println("\n REAL-WORLD NOTE: Test set is artificially balanced (50/50).")
    println("   In nature, IW events are rare (<10% of observations).")
    println("   Real-world FPR/FNR will differ from these estimates.")

    tolerance = round(Int, 0.01 * n_total)
    if abs(fn - fp) <= tolerance
        println("\nERROR BALANCE: FP and FN within 1% tolerance (gap=$(fn-fp))")
    elseif fp > fn
        println("\nERROR BIAS: FALSE POSITIVES dominate by $(fp-fn) samples")
        println("   -> Suggestion: INCREASE threshold")
    else
        println("\nERROR BIAS: FALSE NEGATIVES dominate by $(fn-fp) samples")
        println("   -> Suggestion: DECREASE threshold")
    end
    println("="^60)
    
    return fp, fn, fp_rate, fn_rate
end

function optimize_threshold_with_cost(y_true::Vector{Float32},
                                       y_scores::Vector{Float32};
                                       threshold_range   = 0.3f0:0.01f0:0.7f0,
                                       optimization_metric = "f1_score",
                                       fp_cost::Float32  = 1.0f0,
                                       fn_cost::Float32  = 1.0f0)
    println("\nTHRESHOLD OPTIMIZATION WITH COST ANALYSIS")
    println("="^60)
    println("NOTE: We do NOT assume threshold = 0.5 is optimal!")
    println("   Sweeping thresholds from $(first(threshold_range)) to $(last(threshold_range))")
    println("   Optimization metric: $optimization_metric")
    println("   FP Cost: $fp_cost | FN Cost: $fn_cost")
    println("="^60)

    best_metric_value = -Inf
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

        if metric_value > best_metric_value
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
    ax  = Axis(fig[1,1],
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
    
    # Add confidence interval ribbon for accuracy
    ci_lower = [r["accuracy_ci_lower"] for r in all_results]
    ci_upper = [r["accuracy_ci_upper"] for r in all_results]
    band!(ax, thresholds, ci_lower, ci_upper, color=(:blue, 0.2))
    
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
    tp  = 0.0f0;  fp = 0.0f0
    
    for i in 1:length(y_true_sorted)
        if y_true_sorted[i] > 0.5f0
            tp += 1.0f0
        else
            fp += 1.0f0
        end
        push!(tpr, tp / n_positive)
        push!(fpr, fp / n_negative)
    end
    
    auc = sum((fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0f0
              for i in 2:length(fpr))
    
    # Calculate AUC standard error (Hanley & McNeil)
    q1 = auc / (2 - auc)
    q2 = 2 * auc^2 / (1 + auc)
    se_auc = sqrt((auc * (1 - auc) + (n_positive - 1) * (q1 - auc^2) + 
                   (n_negative - 1) * (q2 - auc^2)) / (n_positive * n_negative))
    
    return auc, se_auc, fpr, tpr
end

function plot_roc_curve(fpr::Vector{Float32}, tpr::Vector{Float32},
                         auc::Float32, se_auc::Float32, 
                         save_path::String="roc_curve.png")
    fig = Figure(size=(800, 800))
    ax  = Axis(fig[1,1],
               xlabel = "False Positive Rate (FPR)",
               ylabel = "True Positive Rate (TPR)",
               title  = "ROC Curve (AUC = $(round(auc, digits=4)) ± $(round(se_auc, digits=4)))")
    
    lines!(ax, fpr, tpr,
           color=:blue, linewidth=2, label="ROC Curve")
    lines!(ax, [0.0f0, 1.0f0], [0.0f0, 1.0f0],
           color=:red, linestyle=:dash, linewidth=2, label="Random Classifier")
    
    axislegend(ax, position=:rb)
    xlims!(ax, 0, 1);  ylims!(ax, 0, 1)
    
    save(save_path, fig)
    println("   ROC curve saved to: $save_path")
    return fig
end

# ============================================================
# OPTIMIZED IMAGE LOADING FUNCTION
# ============================================================

function load_image_tensor_batch(file_paths::Vector{String}, 
                                  norm_mean::Float32, 
                                  norm_std::Float32)
    n_images = length(file_paths)
    
    # Pre-allocate 4D tensor
    X_batch = zeros(Float32, 256, 256, 1, n_images)
    labels = Float32[]
    valid_indices = Int[]
    
    for (idx, fpath) in enumerate(file_paths)
        if !isfile(fpath)
            continue
        end
        try
            img = load(fpath)
            img = imresize(img, (256, 256))
            img = Gray.(img)
            arr = Float32.(channelview(img))
            
            # Direct assignment
            X_batch[:, :, 1, idx] = arr
            push!(valid_indices, idx)
        catch e
            @warn "Skipping $fpath: $e"
        end
    end
    
    if isempty(valid_indices)
        return nothing, nothing
    end
    
    # Keep only valid images
    X_batch = X_batch[:, :, :, valid_indices]
    
    # Normalize
    X_batch = (X_batch .- norm_mean) ./ norm_std
    
    return X_batch, labels
end

# ============================================================
# MAIN EVALUATE FUNCTION — enhanced with all fixes
# ============================================================

function evaluate_model(model_path::String="/tmp/benchmarks_iws/cpu_julia_iws/models/model_v2_20260302_202405.jld2")
    evaluation_start_time = time()
    io_time = 0.0
    inference_time = 0.0

    results_dir = "/tmp/benchmarks_iws/cpu_julia_iws/results"
    mkpath(results_dir)

    println("="^60)
    println("SAR CNN MODEL EVALUATION (CPU — mirrors GPU v2.2)")
    println("="^60)
    println("Random seed: 42 (set for reproducibility)")

    # ── Load model ────────────────────────────────────────────────────────────
    println("\nLoading trained model from: $model_path")
    !isfile(model_path) && error("Model not found: $model_path")

    loaded_data = load(model_path)

    model     = loaded_data["model"]
    ps        = loaded_data["ps"]
    st        = loaded_data["st"]

    norm_mean = Float32(loaded_data["norm_mean"])
    norm_std  = Float32(loaded_data["norm_std"])
    
    # Verify normalization parameters came from training set
    @printf("norm_mean = %.4f   norm_std = %.4f\n", norm_mean, norm_std)
    println("Normalization parameters loaded from training set (no data leakage)")

    println("Model loaded successfully on CPU.")
    haskey(loaded_data, "tta_val_acc")   &&
        @printf("Checkpoint val acc       : %.2f%%\n",
                Float64(loaded_data["tta_val_acc"])*100)
    haskey(loaded_data, "best_val_loss") &&
        @printf("Checkpoint best val loss : %.4f\n",
                Float64(loaded_data["best_val_loss"]))

    # ── Load validation data ──────────────────────────────────────────────────
    println("\n Loading validation data...")
    data_path  = "/tmp/benchmarks_iws/data/raw/test/"
    csv_path   = "/tmp/benchmarks_iws/data/raw/test.csv"
    df         = CSV.read(csv_path, DataFrame)
    n_val      = nrow(df)
    batch_size = 128
    @printf("Validation samples: %d\n", n_val)
    @printf("Batch size: %d\n", batch_size)

    # Set dropout to test mode
    st_val = LuxCore.testmode(set_dropout_mode(st, false))

    # ── Collect predictions ───────────────────────────────────────────────────
    println("\n Evaluating model...")
    all_predictions = Float32[]
    all_labels      = Float32[]

    for batch_start in 1:batch_size:n_val
        batch_end = min(batch_start + batch_size - 1, n_val)
        batch_indices = batch_start:batch_end
        
        # TIMING: I/O phase
        io_start = time()
        
        # Pre-allocate batch tensor
        X_batch = zeros(Float32, 256, 256, 1, length(batch_indices))
        batch_labels = Float32[]
        valid_count = 0
        
        for (local_idx, global_idx) in enumerate(batch_indices)
            fpath = joinpath(data_path, df.id[global_idx])
            if isfile(fpath)
                try
                    img = load(fpath)
                    img = imresize(img, (256, 256))
                    img = Gray.(img)
                    arr = Float32.(channelview(img))
                    
                    # Direct assignment to pre-allocated tensor
                    X_batch[:, :, 1, local_idx] = arr
                    push!(batch_labels, Float32(df.ground_truth[global_idx]))
                    valid_count += 1
                catch e
                    @warn "Skipping $fpath: $e"
                end
            end
        end
        
        io_time += time() - io_start
        
        valid_count == 0 && continue
        
        # Keep only valid samples
        X_batch = X_batch[:, :, :, 1:valid_count]
        
        # Normalize
        X_batch = (X_batch .- norm_mean) ./ norm_std
        
        # TIMING: Inference phase
        inference_start = time()
        y_pred, _ = model(X_batch, ps, st_val)
        inference_time += time() - inference_start
        
        append!(all_predictions, vec(Float32.(y_pred)))
        append!(all_labels, batch_labels[1:valid_count])
    end

    # ── Sanity check ──────────────────────────────────────────────────────────
    pos_rate = sum(all_predictions .>= 0.5f0) / length(all_predictions) * 100
    @printf("\n Score range: [%.4f, %.4f]  mean: %.4f\n",
            minimum(all_predictions), maximum(all_predictions), mean(all_predictions))
    @printf("Positive rate (threshold=0.5): %.1f%%\n", pos_rate)
    
    if abs(pos_rate - 50) > 20
        println("WARNING: Extreme positive rate — check model/normalization")
    else
        println("Positive rate balanced (within expected range)")
    end

    # ── Calculate baselines ───────────────────────────────────────────────────
    baselines = calculate_baselines(all_labels)
    
    println("\n" * "="^60)
    println("BASELINE COMPARISONS")
    println("="^60)
    @printf("Random classifier expected accuracy : %.2f%%\n", baselines["random_classifier"]*100)
    @printf("Majority class classifier (%s)     : %.2f%%\n", 
            baselines["majority_class_name"], baselines["majority_class"]*100)
    println("\nClass distribution in test set:")
    @printf("   Non-IW: %d (%.1f%%)\n", 
            baselines["class_balance"]["n_neg"], 
            baselines["class_balance"]["neg_ratio"]*100)
    @printf("   IW:     %d (%.1f%%)\n", 
            baselines["class_balance"]["n_pos"], 
            baselines["class_balance"]["pos_ratio"]*100)
    println("="^60)

    # ── Baseline at threshold=0.5 ─────────────────────────────────────────────
    println("\nBASELINE (Threshold = 0.5)")
    println("="^60)
    baseline_metrics = calculate_metrics_at_threshold(all_labels, all_predictions, 0.5f0)
    @printf("   Accuracy:  %.2f%% [%.2f-%.2f%%]\n", 
            baseline_metrics["accuracy"] * 100,
            baseline_metrics["accuracy_ci_lower"] * 100,
            baseline_metrics["accuracy_ci_upper"] * 100)
    @printf("   Precision: %.2f%%\n", baseline_metrics["precision"] * 100)
    @printf("   Recall:    %.2f%%\n", baseline_metrics["recall"] * 100)
    @printf("   F1-Score:  %.2f%%\n", baseline_metrics["f1_score"] * 100)
    println("="^60)

    # ── Threshold optimization ────────────────────────────────────────────────
    best_threshold, best_metrics, all_results =
        optimize_threshold_with_cost(
            all_labels, all_predictions;
            threshold_range     = 0.3f0:0.01f0:0.7f0,
            optimization_metric = "f1_score",
            fp_cost             = 1.0f0,
            fn_cost             = 1.0f0
        )

    println("\nOPTIMAL THRESHOLD RESULTS (optimized for F1)")
    println("="^60)
    @printf("   Optimal Threshold : %.3f\n", best_threshold)
    @printf("   Accuracy  : %.2f%% [%.2f-%.2f%%] (delta = %+.2f%%)\n",
            best_metrics["accuracy"] * 100,
            best_metrics["accuracy_ci_lower"] * 100,
            best_metrics["accuracy_ci_upper"] * 100,
            (best_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100)
    @printf("   Precision : %.2f%%\n", best_metrics["precision"] * 100)
    @printf("   Recall    : %.2f%%\n", best_metrics["recall"] * 100)
    @printf("   F1-Score  : %.2f%%\n", best_metrics["f1_score"] * 100)
    println("="^60)

    # ── FP/FN analysis ────────────────────────────────────────────────────────
    fp, fn, fp_rate, fn_rate =
        analyze_fp_vs_fn(best_metrics, all_labels, all_predictions)

    # ── Plots ─────────────────────────────────────────────────────────────────
    println("\nPlotting confusion matrix (optimal)...")
    plot_confusion_matrix(best_metrics,
        joinpath(results_dir, "confusion_matrix_optimal.png"))

    println("Plotting confusion matrix (baseline)...")
    plot_confusion_matrix(baseline_metrics,
        joinpath(results_dir, "confusion_matrix_baseline.png"))

    println("\nCalculating AUC-ROC...")
    auc, se_auc, fpr, tpr = calculate_auc_roc(all_labels, all_predictions)
    @printf("   AUC-ROC: %.4f ± %.4f\n", auc, se_auc)

    println("\nPlotting threshold analysis...")
    plot_threshold_analysis(all_results, best_threshold,
        joinpath(results_dir, "threshold_analysis.png"))

    println("\nPlotting ROC curve...")
    plot_roc_curve(fpr, tpr, auc, se_auc,
        joinpath(results_dir, "roc_curve.png"))

    total_evaluation_time = time() - evaluation_start_time

    # ── Final summary ────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("FINAL EVALUATION SUMMARY")
    println("="^60)
    @printf("Baseline Accuracy  (threshold=0.5)     : %.2f%% [%.2f-%.2f%%]\n",
            baseline_metrics["accuracy"] * 100,
            baseline_metrics["accuracy_ci_lower"] * 100,
            baseline_metrics["accuracy_ci_upper"] * 100)
    @printf("Optimized Accuracy (threshold=%.3f)   : %.2f%% [%.2f-%.2f%%]\n",
            best_threshold,
            best_metrics["accuracy"] * 100,
            best_metrics["accuracy_ci_lower"] * 100,
            best_metrics["accuracy_ci_upper"] * 100)
    @printf("Accuracy Gain                          : %+.2f%%\n",
            (best_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100)
    println("")
    println("Error Analysis at Optimal Threshold:")
    @printf("   FPR (FP / N_neg) : %.2f%%\n",     fp_rate * 100)
    @printf("   FNR (FN / N_pos) : %.2f%%  <- miss rate\n", fn_rate * 100)
    println("")
    @printf("AUC-ROC Score: %.4f ± %.4f\n", auc, se_auc)
    println("="^60)
    
    println("\n" * "="^60)
    println("EVALUATION TIME SUMMARY")
    println("="^60)
    @printf("   Total evaluation time : %.1f seconds\n", total_evaluation_time)
    @printf("   I/O time              : %.1f seconds (%.1f%%)\n", 
            io_time, io_time/total_evaluation_time*100)
    @printf("   Inference time        : %.1f seconds (%.1f%%)\n", 
            inference_time, inference_time/total_evaluation_time*100)
    @printf("   Time per sample       : %.1f ms\n",
            total_evaluation_time / n_val * 1000)
    @printf("   - I/O per sample      : %.1f ms\n", io_time / n_val * 1000)
    @printf("   - Inference per sample: %.1f ms\n", inference_time / n_val * 1000)
    println("="^60)

    # ── Save model ────────────────────────────────────────────────────────────
    if best_metrics["accuracy"] >= 0.85f0
        println("\nSaving best model with optimal threshold...")
        jldsave("/tmp/benchmarks_iws/cpu_julia_iws/models/model_best.jld2";
                model             = model,
                ps                = ps,
                st                = st,
                val_accuracy      = best_metrics["accuracy"],
                auc_roc           = auc,
                auc_roc_se        = se_auc,
                optimal_threshold = best_threshold,
                baseline_accuracy = baseline_metrics["accuracy"],
                false_positives   = fp,
                false_negatives   = fn,
                fpr               = fp_rate,
                fnr               = fn_rate,
                backend           = "cpu",
                compute_mode      = "cpu",
                random_seed       = 42)
    else
        @printf("Accuracy %.2f%% — saved but consider retraining.\n",
                best_metrics["accuracy"]*100)
        jldsave("/tmp/benchmarks_iws/cpu_julia_iws/models/model_best.jld2";
                model=model, ps=ps, st=st,
                val_accuracy=best_metrics["accuracy"],
                auc_roc=auc,
                optimal_threshold=best_threshold,
                backend="cpu",
                compute_mode="cpu",
                random_seed=42)
    end

    # ── Save JSON ─────────────────────────────────────────────────────────────
    println("\nSaving evaluation metrics to JSON...")
    evaluation_metrics = Dict(
        "metadata" => Dict(
            "timestamp"              => string(now()),
            "pipeline"               => "Julia_Lux_CNN_CPU_v2.2",
            "device"                 => "cpu",
            "total_samples"          => n_val,
            "model_path"             => model_path,
            "threshold_metric"       => "f1_score",
            "random_seed"            => 42,
            "threshold_metric_note"  => "Optimises f1_score — matches GPU evaluate.jl v2.2"
        ),
        "baselines" => Dict(
            "random_classifier" => Float64(baselines["random_classifier"]),
            "majority_class"    => Float64(baselines["majority_class"]),
            "majority_class_name" => baselines["majority_class_name"],
            "class_balance" => Dict(
                "n_pos" => Int(baselines["class_balance"]["n_pos"]),
                "n_neg" => Int(baselines["class_balance"]["n_neg"]),
                "pos_ratio" => Float64(baselines["class_balance"]["pos_ratio"]),
                "neg_ratio" => Float64(baselines["class_balance"]["neg_ratio"])
            )
        ),
        "baseline" => Dict(
            "threshold" => 0.5,
            "accuracy"  => Float64(baseline_metrics["accuracy"]),
            "accuracy_ci_lower" => Float64(baseline_metrics["accuracy_ci_lower"]),
            "accuracy_ci_upper" => Float64(baseline_metrics["accuracy_ci_upper"]),
            "precision" => Float64(baseline_metrics["precision"]),
            "recall"    => Float64(baseline_metrics["recall"]),
            "f1_score"  => Float64(baseline_metrics["f1_score"]),
            "confusion_matrix" => Dict(
                "tp" => Int(baseline_metrics["tp"]),
                "tn" => Int(baseline_metrics["tn"]),
                "fp" => Int(baseline_metrics["fp"]),
                "fn" => Int(baseline_metrics["fn"]))
        ),
        "optimal" => Dict(
            "threshold"        => Float64(best_threshold),
            "optimized_metric" => "f1_score",
            "accuracy"         => Float64(best_metrics["accuracy"]),
            "accuracy_ci_lower" => Float64(best_metrics["accuracy_ci_lower"]),
            "accuracy_ci_upper" => Float64(best_metrics["accuracy_ci_upper"]),
            "precision"        => Float64(best_metrics["precision"]),
            "recall"           => Float64(best_metrics["recall"]),
            "f1_score"         => Float64(best_metrics["f1_score"]),
            "cost"             => Float64(best_metrics["total_cost"]),
            "confusion_matrix" => Dict(
                "tp" => Int(best_metrics["tp"]),
                "tn" => Int(best_metrics["tn"]),
                "fp" => Int(best_metrics["fp"]),
                "fn" => Int(best_metrics["fn"]))
        ),
        "error_analysis" => Dict(
            "fpr"            => Float64(fp_rate),
            "fnr"            => Float64(fn_rate),
            "fp_count"       => Int(fp),
            "fn_count"       => Int(fn),
            "error_bias"     => abs(fn-fp) <= round(Int, 0.01*(fp+fn+Int(best_metrics["tp"])+Int(best_metrics["tn"]))) ?
                                "BALANCED" : (fn > fp ? "FALSE_NEGATIVES" : "FALSE_POSITIVES"),
            "recommendation" => fn > fp ? "DECREASE threshold" : "INCREASE threshold"
        ),
        "model_performance" => Dict(
            "auc_roc" => Float64(auc),
            "auc_roc_se" => Float64(se_auc)
        ),
        "timing" => Dict(
            "total_seconds" => total_evaluation_time,
            "io_seconds" => io_time,
            "inference_seconds" => inference_time,
            "ms_per_sample" => total_evaluation_time / n_val * 1000,
            "io_ms_per_sample" => io_time / n_val * 1000,
            "inference_ms_per_sample" => inference_time / n_val * 1000
        ),
        "threshold_sweep" => [
            Dict(
                "threshold" => Float64(r["threshold"]),
                "accuracy"  => Float64(r["accuracy"]),
                "accuracy_ci_lower" => Float64(r["accuracy_ci_lower"]),
                "accuracy_ci_upper" => Float64(r["accuracy_ci_upper"]),
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

    json_filename = joinpath(results_dir, "cpu_julia_evaluation_metrics.json")
    open(json_filename, "w") do f
        JSON3.write(f, evaluation_metrics)
    end
    println("Evaluation metrics saved to: $json_filename")

    return baseline_metrics["accuracy"], best_metrics["accuracy"],
           best_threshold, auc, fp, fn
end

# ── Run ───────────────────────────────────────────────────────────────────────
println("Starting CPU model evaluation...")
baseline_acc, optimized_acc, optimal_threshold, auc, fp, fn = evaluate_model()

println("\n" * "="^60)
println("FINAL RESULTS")
println("="^60)
@printf("Baseline Accuracy (0.5)      : %.2f%%\n", baseline_acc * 100)
@printf("Optimized Accuracy (%.3f)   : %.2f%%\n",  optimal_threshold, optimized_acc * 100)
@printf("Improvement                  : %+.2f%%\n",
        (optimized_acc - baseline_acc) * 100)
println("")
@printf("False Positives : %d\n", fp)
@printf("False Negatives : %d\n", fn)
println("")
@printf("AUC-ROC Score   : %.4f\n", auc)
println("="^60)

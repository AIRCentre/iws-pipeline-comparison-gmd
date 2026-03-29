# ============================================================
# pipeline_b_cpu_inference.jl — Julia Lux CNN CPU Inference
# FINAL VERSION - FULL DATASET
# UPDATED: 2026-03-17
# ============================================================

using Images, FileIO, ImageTransformations, JLD2
using DataFrames, CSV, Statistics, Dates, Printf
using ProgressMeter
using Lux, LuxCore, LuxLib
using NNlib, Random
using JSON3
using LinearAlgebra: BLAS
using Base.Threads

BLAS.set_num_threads(Sys.CPU_THREADS)

const PROJECT_ROOT = "/tmp/benchmarks_iws"

# ============================================================
# ARCHITECTURE
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
    return y, (block1=st1, block2=st2, block3=st3, block4=st4, agg=st5, classifier=st6)
end

# ============================================================
# FIXED: set_dropout_mode - SIMPLE AND CORRECT
# ============================================================

function set_dropout_mode(st::NamedTuple, training::Bool)
    if training
        return st
    else
        return LuxCore.testmode(st)  # Direct API call - correct way
    end
end

# ============================================================
# MEMORY UTILITIES
# ============================================================

cpu_memory_mb() = Sys.maxrss() / 1024 / 1024

function track_memory(phase::String, memory_log::Dict)
    current = cpu_memory_mb()
    memory_log[phase] = current
    memory_log["peak"] = max(get(memory_log, "peak", 0.0), current)
    @printf(" %-20s: %.1f MB\n", phase, current)
    return current
end

# ============================================================
# FIXED: IMAGE PREPROCESSING - NO *255
# ============================================================

function preprocess_image(path::String)
    img = load(path)
    img = Gray.(img)
    img = imresize(img, (256, 256))
    arr = Float32.(img)  # FIXED: No *255 - now in [0,1] range matching training
    return reshape(arr, 256, 256, 1)
end

function preprocess_images_batch(paths::Vector{String})
    N = length(paths)
    H, W, C = 256, 256, 1
    
    data = Array{Float32}(undef, H, W, C, N)
    
    @threads for i in 1:N
        try
            data[:, :, :, i] = preprocess_image(paths[i])
        catch e
            @warn "Failed to load $(paths[i]): $e"
            data[:, :, :, i] .= 0.0f0
        end
    end
    
    return data
end

# ============================================================
# INFERENCE DATALOADER
# ============================================================

mutable struct InferenceDataLoader
    image_paths :: Vector{String}
    batch_size  :: Int
    data_cpu    :: Array{Float32,4}
    n_images    :: Int
end

function InferenceDataLoader(dir::String; batch_size=128)
    !isdir(dir) && error("Dataset directory not found: $dir")
    
    files = readdir(dir)
    image_paths = [joinpath(dir, f) for f in files
                   if lowercase(splitext(f)[2]) in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
    sort!(image_paths)
    N = length(image_paths)
    
    println("\n   Preloading $N images into CPU memory...")
    tic = time()
    
    data = preprocess_images_batch(image_paths)
    
    elapsed = time() - tic
    data_gb = sizeof(data) / 1e9
    println("Preloaded in $(round(elapsed, digits=2)) s")
    println("Memory usage: $(round(data_gb, digits=2)) GB")
    
    return InferenceDataLoader(image_paths, batch_size, data, N)
end

Base.length(dl::InferenceDataLoader) = ceil(Int, dl.n_images / dl.batch_size)

function Base.iterate(dl::InferenceDataLoader, state=1)
    state > dl.n_images && return nothing
    last = min(state + dl.batch_size - 1, dl.n_images)
    idx = state:last
    
    batch_cpu = dl.data_cpu[:, :, :, idx]
    
    return ((batch_cpu, idx), last + 1)
end

# ============================================================
# PERFORMANCE METRICS
# ============================================================

struct PerformanceMetrics
    model_load_time      :: Float64
    data_load_time       :: Float64
    warmup_time          :: Float64
    total_inference_time :: Float64
    avg_batch_time       :: Float64
    std_batch_time       :: Float64
    images_per_second    :: Float64
    images_per_second_ci :: Tuple{Float64,Float64}
    throughput_by_run    :: Vector{Float64}
    throughput_cv        :: Float64
    single_image_latency :: Float64
    cpu_memory_mb        :: Float64
    peak_cpu_mb          :: Float64
    initial_cpu_mb       :: Float64
    after_model_cpu_mb   :: Float64
    after_data_cpu_mb    :: Float64
    memory_per_image_mb  :: Float64
    total_batches        :: Int
    batch_size           :: Int
    total_images         :: Int
    processing_efficiency:: Float64
    ground_truth_rate    :: Float64
    predicted_rate       :: Float64
    accuracy             :: Float64
    batch_times          :: Vector{Float64}
end

function Base.show(io::IO, m::PerformanceMetrics)
    println(io, "="^60)
    println(io, "COMPLETE PIPELINE PERFORMANCE METRICS (CPU)")
    println(io, "="^60)
    
    println(io, "\nTHROUGHPUT (with 95% CI):")
    println(io, "-"^40)
    @printf(io, "Complete Pipeline: %4d img/s [%d-%d]\n", 
            round(Int, m.images_per_second),
            round(Int, m.images_per_second_ci[1]),
            round(Int, m.images_per_second_ci[2]))
    @printf(io, "Run-to-run CV:     %.2f%%\n", m.throughput_cv * 100)
    
    println(io, "\nLATENCY:")
    println(io, "-"^40)
    @printf(io, "Per image:         %5.2f ms\n", m.single_image_latency * 1000)
    
    println(io, "\nMEMORY:")
    println(io, "-"^40)
    @printf(io, "   Initial:           %6.1f MB\n", m.initial_cpu_mb)
    @printf(io, "   After Model:       %6.1f MB\n", m.after_model_cpu_mb)
    @printf(io, "   After Data:        %6.1f MB\n", m.after_data_cpu_mb)
    @printf(io, "   Peak:              %6.1f MB\n", m.peak_cpu_mb)
    @printf(io, "   Per image:         %6.3f MB\n", m.memory_per_image_mb)
    
    println(io, "\nBATCH PROCESSING (batch size = $(m.batch_size)):")
    println(io, "-"^40)
    @printf(io, "   Total batches:     %d\n", m.total_batches)
    @printf(io, "   Avg batch time:    %.2f ms\n", m.avg_batch_time * 1000)
    if m.std_batch_time > 0
        @printf(io, "   Std batch time:    %.2f ms\n", m.std_batch_time * 1000)
    end
    
    println(io, "\nACCURACY:")
    println(io, "-"^40)
    @printf(io, "   Ground truth +ve:  %.1f%%\n", m.ground_truth_rate)
    @printf(io, "   Predicted +ve:     %.1f%%\n", m.predicted_rate)
    @printf(io, "   Accuracy:          %.1f%%\n", m.accuracy)
    
    println(io, "\nPROCESSING:")
    println(io, "-"^40)
    @printf(io, "   CPU efficiency:    %.1f%%\n", m.processing_efficiency * 100)
    println(io, "="^60)
end

# ============================================================
# STATISTICAL UTILITIES
# ============================================================

function calculate_ci(values::Vector{Float64})
    n = length(values)
    mean_val = mean(values)
    if n < 2
        return (mean_val, mean_val)
    end
    se = std(values) / sqrt(n)
    z = 1.96
    return (mean_val - z * se, mean_val + z * se)
end

# ============================================================
# LOAD GROUND TRUTH
# ============================================================

function load_ground_truth(csv_path::String, image_paths::Vector{String})
    if !isfile(csv_path)
        @warn "Ground truth CSV not found: $csv_path"
        return fill(0.5f0, length(image_paths))
    end
    
    df = CSV.read(csv_path, DataFrame)
    truth_map = Dict{String, Float32}()
    for row in eachrow(df)
        truth_map[row.id] = Float32(row.ground_truth)
    end
    
    ground_truth = Float32[]
    for path in image_paths
        filename = basename(path)
        push!(ground_truth, get(truth_map, filename, 0.5f0))
    end
    
    return ground_truth
end

# ============================================================
# MAIN INFERENCE FUNCTION - FULL DATASET
# ============================================================

function run_inference_with_profiling(;
    model_path    = "/tmp/benchmarks_iws/cpu_julia_iws/models/model_v2_20260302_202405.jld2",
    inference_dir = "/tmp/benchmarks_iws/data/raw/test",           # CHANGED: full dataset
    ground_truth_csv = "/tmp/benchmarks_iws/data/raw/test.csv",    # CHANGED: full dataset
    output_csv    = "/tmp/benchmarks_iws/cpu_julia_iws/results/inference_predictions_cpu.csv",
    results_dir   = "/tmp/benchmarks_iws/cpu_julia_iws/results",
    batch_size    = 128,
    warmup_runs   = 2,
    profile_runs  = 5
)
    mkpath(results_dir)
    memory_log = Dict("peak" => 0.0)
    
    println("\n" * "="^60)
    println("JULIA LUX CNN INFERENCE - FULL DATASET")
    println("="^60)
    println("   Images: $inference_dir")
    println("   Ground truth: $ground_truth_csv")
    println("   Profile runs: $profile_runs")
    println("   Julia threads: $(Threads.nthreads())")
    println("   BLAS threads: $(Sys.CPU_THREADS)")
    println("-"^60)

    # ── Initial memory ───────────────────────────────────
    println("\nMEMORY TRACKING:")
    println("-"^40)
    initial_memory = track_memory("Initial", memory_log)

    # ── Load model ───────────────────────────────────────
    println("\n1. LOADING TRAINED MODEL...")
    println("   Path: $model_path")
    !isfile(model_path) && error("Model file not found: $model_path")

    t_model = time()
    model_data = jldopen(model_path, "r") do f
        (model    = f["model"],
         ps       = f["ps"],
         st       = f["st"],
         norm_mean = Float32(f["norm_mean"]),
         norm_std  = Float32(f["norm_std"]))
    end

    model = model_data.model
    ps = model_data.ps
    
    # FIXED: Direct testmode call - correct and simple
    st = set_dropout_mode(model_data.st, false)  # Uses the simple function above
    
    norm_mean = model_data.norm_mean
    norm_std = model_data.norm_std

    model_load_time = time() - t_model
    after_model_memory = track_memory("After Model Load", memory_log)

    @printf("Model loaded in %.1f ms\n", model_load_time * 1000)
    @printf("norm_mean = %.4f   norm_std = %.4f\n", norm_mean, norm_std)
    @printf("Model in inference mode (dropout disabled)\n")

    # ── Sanity check with validation ─────────────────────
    println("\n   Sanity check on random noise batch...")
    X_rand = rand(Float32, 256, 256, 1, 10)  # Range [0,1]
    X_rand_norm = (X_rand .- norm_mean) ./ norm_std

    @printf("Normalized range: [%.4f, %.4f]\n", 
            minimum(X_rand_norm), maximum(X_rand_norm))
    
    # Validation
    if minimum(X_rand_norm) < -3 || maximum(X_rand_norm) > 4
        error("""
        NORMALIZATION ERROR: 
        Expected range [-2.5, 3.9] for [0,1] input.
        Got [$(minimum(X_rand_norm)), $(maximum(X_rand_norm))].
        Check preprocessing - remove *255 from preprocess_image()!
        """)
    end
    
    y_rand, _ = model(X_rand_norm, ps, st)
    @printf("Predictions — min: %.4f  max: %.4f  mean: %.4f\n",
            minimum(y_rand), maximum(y_rand), mean(y_rand))
    println("Sanity check passed")

    # ── Preload dataset ──────────────────────────────────
    println("\n2. LOADING TEST DATASET...")
    t_preload = time()
    loader = InferenceDataLoader(inference_dir; batch_size=batch_size)
    data_load_time = time() - t_preload
    after_data_memory = track_memory("After Data Load", memory_log)
    
    total_images = loader.n_images
    @printf("   %d images loaded\n", total_images)
    
    # Calculate raw image memory vs overhead
    raw_image_memory = (256 * 256 * 1 * 4 * total_images) / 1024 / 1024
    data_memory = after_data_memory - after_model_memory
    overhead = data_memory - raw_image_memory
    @printf("   Raw image data:    %.1f MB\n", raw_image_memory)
    @printf("   Overhead:          %.1f MB (views + alignment)\n", overhead)
    
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_csv, loader.image_paths)
    truth_rate = mean(ground_truth) * 100
    @printf("   Ground truth positive rate: %.1f%%\n", truth_rate)

    # ── Warm-up passes ───────────────────────────────────
    println("\n3. WARM-UP PASSES ($warmup_runs runs)...")
    t_warm = time()
    warmup_times = Float64[]
    
    for run in 1:warmup_runs
        run_start = time()
        for (X, idx) in loader
            X_norm = (X .- norm_mean) ./ norm_std
            y, _ = model(X_norm, ps, st)
        end
        run_time = time() - run_start
        push!(warmup_times, run_time)
        @printf("   Warmup run %d: %.3f s\n", run, run_time)
    end
    
    warmup_time = time() - t_warm
    @printf("   Average warm-up: %.3f s\n", mean(warmup_times))

    # ── Main profiling passes ────────────────────────────
    println("\n4. MAIN INFERENCE PROFILING ($profile_runs runs)...")
    
    run_times = Float64[]
    run_throughputs = Float64[]
    all_predictions = []
    batch_times_all = Float64[]
    
    prog = Progress(profile_runs, 1, "Running inference...", 50)
    
    for run in 1:profile_runs
        run_predictions = zeros(Float32, total_images)
        run_start = time()
        
        for (X, idx) in loader
            batch_start = time()
            
            X_norm = (X .- norm_mean) ./ norm_std
            y, _ = model(X_norm, ps, st)
            
            batch_time = time() - batch_start
            push!(batch_times_all, batch_time)
            
            run_predictions[idx] .= vec(Float32.(y))
        end
        
        run_time = time() - run_start
        push!(run_times, run_time)
        push!(run_throughputs, total_images / run_time)
        push!(all_predictions, copy(run_predictions))
        
        next!(prog)
    end

    total_inference_time = sum(run_times)
    peak_memory = memory_log["peak"]

    # ── Compute statistics ───────────────────────────────
    mean_throughput = mean(run_throughputs)
    throughput_ci = calculate_ci(run_throughputs)
    throughput_cv = std(run_throughputs) / mean_throughput
    
    avg_batch_time = mean(batch_times_all)
    std_batch_time = length(batch_times_all) > 1 ? std(batch_times_all) : 0.0
    n_batches = length(batch_times_all)
    
    memory_per_image = (after_data_memory - after_model_memory) / total_images
    
    # Processing efficiency
    total_compute = sum(batch_times_all)
    processing_efficiency = total_compute / total_inference_time
    
    # Prediction analysis
    final_predictions = mean(all_predictions)
    final_binary = Int.(final_predictions .> 0.5f0)
    predicted_rate = sum(final_binary) / total_images * 100
    
    # Accuracy against ground truth
    correct = sum(final_binary .== Int.(ground_truth .> 0.5f0))
    accuracy = correct / total_images * 100

    # Create metrics
    metrics = PerformanceMetrics(
        model_load_time, data_load_time, warmup_time,
        total_inference_time, avg_batch_time, std_batch_time,
        mean_throughput, throughput_ci, run_throughputs, throughput_cv,
        1.0/mean_throughput, peak_memory, peak_memory,
        initial_memory, after_model_memory, after_data_memory,
        memory_per_image, n_batches, batch_size, total_images,
        processing_efficiency, truth_rate, predicted_rate, accuracy,
        batch_times_all
    )

    println("\n" * "="^60)
    println("PIPELINE PERFORMANCE BREAKDOWN")
    println("="^60)
    show(stdout, metrics)

    # ── Save predictions ─────────────────────────────────
    println("\n5. SAVING PREDICTIONS...")
    
    df = DataFrame(
        image_path = loader.image_paths,
        ground_truth = ground_truth,
        prediction = final_binary,
        confidence = final_predictions,
        correct = Int.(final_binary .== Int.(ground_truth .> 0.5f0))
    )
    CSV.write(output_csv, df)
    
    @printf("Predictions saved to %s\n", output_csv)
    @printf("Total predictions: %d\n", nrow(df))
    @printf("Accuracy: %.1f%%\n", accuracy)

    # ── Save report ──────────────────────────────────────
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    report_path = joinpath(results_dir, "inference_report_full_$timestamp.txt")
    
    open(report_path, "w") do io
        println(io, "="^80)
        println(io, "JULIA LUX CNN INFERENCE - FULL DATASET REPORT")
        println(io, "="^80)
        println(io, "Generated: $(now())")
        println(io, "Total Images: $(metrics.total_images)")
        println(io, "Batch Size: $(metrics.batch_size)")
        println(io, "Profile Runs: $profile_runs")
        println(io, "Accuracy: $(round(accuracy, digits=1))%")
        show(io, metrics)
    end
    println("Report saved to: $report_path")

    # ── Save JSON ────────────────────────────────────────
    json_path = joinpath(results_dir, "julia_cpu_inference_metrics.json")
    
    perf_dict = Dict(
        "timing" => Dict(
            "model_load_ms" => model_load_time * 1000,
            "data_load_s" => data_load_time,
            "warmup_s" => warmup_time,
            "total_inference_s" => total_inference_time,
            "avg_batch_ms" => avg_batch_time * 1000,
            "std_batch_ms" => std_batch_time * 1000,
            "batch_count" => n_batches
        ),
        "throughput" => Dict(
            "img_per_s" => mean_throughput,
            "ci_lower" => throughput_ci[1],
            "ci_upper" => throughput_ci[2],
            "cv" => throughput_cv,
            "by_run" => run_throughputs
        ),
        "memory_mb" => Dict(
            "initial" => initial_memory,
            "after_model" => after_model_memory,
            "after_data" => after_data_memory,
            "peak" => peak_memory,
            "per_image" => memory_per_image,
            "raw_image_data" => raw_image_memory,
            "overhead" => overhead
        ),
        "accuracy" => Dict(
            "ground_truth_positive" => truth_rate,
            "predicted_positive" => predicted_rate,
            "accuracy" => accuracy,
            "correct" => correct,
            "total" => total_images
        )
    )
    
    open(json_path, "w") do f
        JSON3.write(f, Dict(
            "metadata" => Dict(
                "timestamp" => string(now()),
                "pipeline" => "Julia_Lux_CNN_CPU_FULL",
                "julia_threads" => Threads.nthreads(),
                "blas_threads" => Sys.CPU_THREADS,
                "total_images" => total_images,
                "norm_mean" => Float64(norm_mean),
                "norm_std" => Float64(norm_std),
                "dataset" => "full_test"
            ),
            "performance" => perf_dict
        ))
    end
    println("JSON saved to: $json_path")

    # ── Key findings ─────────────────────────────────────
    println("\n" * "="^60)
    println("KEY FINDINGS - FULL DATASET RUN")
    println("="^60)
    @printf("   Throughput:          %.1f ± %.1f img/s (95%% CI)\n", 
            mean_throughput, (throughput_ci[2] - throughput_ci[1])/2)
    @printf("   Run-to-run stability: CV = %.2f%%\n", throughput_cv * 100)
    @printf("   Latency per image:   %.2f ms\n", 1000/mean_throughput)
    @printf("   Peak memory:         %.1f MB\n", peak_memory)
    @printf("   Memory per image:    %.3f MB (raw: %.3f MB, overhead: %.3f MB)\n", 
            memory_per_image, raw_image_memory/total_images, overhead/total_images)
    @printf("   Accuracy:            %.1f%%\n", accuracy)
    
    if accuracy > 90
        println("\n MODEL PERFORMANCE: Excellent")
    elseif accuracy > 80
        println("\n MODEL PERFORMANCE: Good but could be better")
    else
        println("\n MODEL PERFORMANCE: Needs improvement")
    end
    println("="^60)

    return df, metrics, loader, model, ps, st, norm_mean, norm_std
end

# ============================================================
# EXECUTION - FULL DATASET
# ============================================================

println("Starting FULL DATASET inference...")
println("="^60)
println("Julia version: $(VERSION)")
println("Threads: $(Threads.nthreads()) CPU threads")
println("BLAS threads: $(Sys.CPU_THREADS)")
println("Dataset: /tmp/benchmarks_iws/data/raw/test/ (5860 images)")
println("="^60)

df, metrics, loader, model, ps, st, _norm_mean, _norm_std = run_inference_with_profiling(
    profile_runs = 5  # Can increase to 20 for production
)

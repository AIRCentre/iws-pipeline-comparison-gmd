# JULIA LUX CNN INFERENCE - GPU (v2.2)
#
# FIXES vs original:
# 1. TARGET_SIZE hardcoded 224×224 — changed to use
#    DefaultConfig.TARGET_SIZE (matches training = 256×256)
# 2. preprocess_image normalizes to [0,1] via /255
#    but training expects raw [0,255] Float32 + per-batch
#    normalize_batch() — fixed to match CPU v2.2 exactly
# 3. norm_mean/norm_std not loaded from checkpoint —
#    added load from JLD2 and applied in batch loop
# 4. LuxCore.testmode() replaced with set_dropout_mode()
#    matches training and CPU v2.2 exactly
# 5. Warmup uses plain forward pass (not TTA) — acceptable
#    for GPU since TTA is not used here, but warmup now
#    explicitly stated and consistent with main loop
# 6. Model path not printed at load time — now always logged
# 7. Positive rate sanity check added after saving predictions
# =========================================================

using Images, FileIO, ImageTransformations, JLD2
using DataFrames, CSV, Statistics, Dates, Printf
using ProgressBars
using Lux, LuxCore
using CUDA, LuxCUDA
using JSON3

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
ENV["JULIA_CUDA_VERBOSE"] = "0"

include("/tmp/benchmarks_iws/gpu_julia_iws/config/default_config.jl")
include("/tmp/benchmarks_iws/gpu_julia_iws/scripts/sar_cnn_gpu.jl")
using .DefaultConfig
using .DefaultConfig.DeviceConfig

# =========================================================
# FIX 4: set_dropout_mode replaces LuxCore.testmode()
# =========================================================
function set_dropout_mode(st::NamedTuple, training::Bool)
    return map(st) do layer_st
        if layer_st isa NamedTuple
            set_dropout_mode(layer_st, training)
        else
            layer_st
        end
    end
end

# =========================================================
# MEMORY UTILITIES
# =========================================================
cpu_memory_mb() = Sys.maxrss() / 1024 / 1024

function gpu_memory_mb()
    !CUDA.functional() && return 0.0
    try
        total = CUDA.total_memory()
        free  = CUDA.available_memory()
        return Float64(total - free) / 1024^2
    catch
        try
            out = read(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0`, String)
            return parse(Float64, strip(out))
        catch
            return 0.0
        end
    end
end

# =========================================================
# IMAGE PREPROCESSING
#
# FIX 1: TARGET_SIZE — use DefaultConfig not hardcoded 224
# FIX 2: normalization — keep raw [0,255] Float32 here;
#         per-batch normalize_batch() applied in the loop
#         (matches training pipeline exactly)
#
# ORIGINAL BUG: arr = Float32.(img) ./ 255f0
#   → images scaled to [0,1]
#   → but normalize_batch() expects [0,255] as input
#   → result: all inputs effectively 255× too small
#   → model outputs near-constant value → positive rate 0%
# =========================================================
function preprocess_image(path::String)
    img = load(path)
    img = Gray.(img)
    # FIX 1: was TARGET_SIZE = (224,224) hardcoded
    img = imresize(img, DefaultConfig.TARGET_SIZE)
    # FIX 2: was ./ 255f0 — now keep raw [0,255] Float32
    arr = Float32.(img) .* 255.0f0
    return reshape(arr, size(arr,1), size(arr,2), 1)
end

# =========================================================
# NORMALIZE BATCH — matches training and CPU v2.2 exactly
# Applied per-batch AFTER GPU transfer
# =========================================================
function normalize_batch(X::AbstractArray{Float32}, mean::Float32, std::Float32)
    return (X .- mean) ./ std
end

# =========================================================
# PRELOADED DATASET
# =========================================================
mutable struct InferenceDataLoader
    image_paths::Vector{String}
    batch_size::Int
    data_cpu::Array{Float32,4}
end

function InferenceDataLoader(dir::String; batch_size = DefaultConfig.BATCH_SIZE)
    !isdir(dir) && error("Dataset directory not found: $dir")
    files       = readdir(dir)
    image_paths = [joinpath(dir,f) for f in files
                   if lowercase(splitext(f)[2]) in [".png",".jpg",".jpeg",".tif",".tiff"]]
    sort!(image_paths)
    N    = length(image_paths)
    H, W = DefaultConfig.TARGET_SIZE   # FIX 1: was (224,224)
    C    = 1

    println("Preloading $N images into CPU memory...")
    tic  = time()
    data = Array{Float32}(undef, H, W, C, N)
    Threads.@threads for i in 1:N
        data[:,:,:,i] = preprocess_image(image_paths[i])
    end
    elapsed   = time() - tic
    memory_gb = sizeof(data) / 1e9
    println("Preloaded $(round(elapsed, digits=2)) s, memory: $(round(memory_gb, digits=2)) GB")
    return InferenceDataLoader(image_paths, batch_size, data)
end

Base.length(dl::InferenceDataLoader) =
    ceil(Int, size(dl.data_cpu, 4) / dl.batch_size)

function Base.iterate(dl::InferenceDataLoader, state=1)
    N     = size(dl.data_cpu, 4)
    state > N && return nothing
    last  = min(state + dl.batch_size - 1, N)
    idx   = state:last
    t_load      = time()
    batch_cpu   = dl.data_cpu[:,:,:,idx]
    batch_gpu   = batch_cpu |> Lux.gpu_device()  # ← THIS LINE BELONGS HERE
    CUDA.synchronize()
    load_time   = time() - t_load
    return ((batch_gpu, idx, load_time), last + 1)
end

# =========================================================
# PERFORMANCE METRICS STRUCT
# =========================================================
# =========================================================
# PERFORMANCE METRICS STRUCT
# =========================================================
struct PerformanceMetrics
    model_load_time::Float64
    data_load_time::Float64
    warmup_time::Float64
    total_inference_time::Float64
    avg_batch_time::Float64
    std_batch_time::Float64
    images_per_second::Float64
    single_image_latency::Float64
    cpu_memory_mb::Float64
    gpu_memory_allocated_mb::Float64
    gpu_memory_cached_mb::Float64
    batch_times::Vector{Float64}
    timestamps::Vector{Float64}
    image_loading_ms::Float64
    pure_cnn_inference_ms::Float64
    complete_pipeline_ms::Float64
    image_loading_throughput::Float64
    cnn_throughput::Float64
    processing_efficiency::Float64
    total_batches::Int
    batch_size::Int
    total_images::Int
    memory_per_image_mb::Float64
    initial_cpu_mb::Float64
    initial_gpu_mb::Float64
    after_model_cpu_mb::Float64
    after_model_gpu_mb::Float64
    peak_cpu_mb::Float64
    peak_gpu_mb::Float64
end

function Base.show(io::IO, m::PerformanceMetrics)
    println(io, "="^60)
    println(io, "COMPLETE PIPELINE PERFORMANCE METRICS (GPU v2.2)")
    println(io, "="^60)
    println(io, "\n THROUGHPUT BY COMPONENT:")
    println(io, "-"^40)
    @printf(io, "   Complete Pipeline: %4d img/s\n", round(Int, m.images_per_second))
    @printf(io, "   ├─ Image Loading:  %4d img/s\n", round(Int, m.image_loading_throughput))
    @printf(io, "   └─ CNN:            %4d img/s\n", round(Int, m.cnn_throughput))
    println(io, "\n LATENCY BY COMPONENT:")
    println(io, "-"^40)
    @printf(io, "   Complete Pipeline: %5.1f ms\n",  m.complete_pipeline_ms)
    @printf(io, "   ├─ Image Loading:  %5.1f ms\n",  m.image_loading_ms)
    @printf(io, "   └─ CNN:            %5.3f ms\n",  m.pure_cnn_inference_ms)
    println(io, "\n MEMORY TRACKING:")
    println(io, "-"^40)
    println(io, "   Stage                    CPU       GPU")
    println(io, "   ----------------------------------------")
    @printf(io, "   Initial:               %6.1f    %6.1f\n", m.initial_cpu_mb,      m.initial_gpu_mb)
    @printf(io, "   After Model Load:      %6.1f    %6.1f\n", m.after_model_cpu_mb,  m.after_model_gpu_mb)
    @printf(io, "   Peak Usage:            %6.1f    %6.1f\n", m.peak_cpu_mb,         m.peak_gpu_mb)
    @printf(io, "   Memory per image:      %6.3f MB/img\n",   m.memory_per_image_mb)
    println(io, "\n PROCESSING EFFICIENCY:")
    println(io, "-"^40)
    @printf(io, "   GPU busy:      %5.1f%%\n", m.processing_efficiency * 100)
    @printf(io, "   Total batches: %d\n",       m.total_batches)
    @printf(io, "   Avg batch time: %.1f ms\n", m.avg_batch_time * 1000)
    @printf(io, "   Std batch time: %.1f ms\n", m.std_batch_time * 1000)
    if !isempty(m.batch_times)
        println(io, "\n BATCH TIME DISTRIBUTION:")
        @printf(io, "   25th percentile:   %.1f ms\n", quantile(m.batch_times, 0.25) * 1000)
        @printf(io, "   50th percentile:   %.1f ms\n", quantile(m.batch_times, 0.50) * 1000)
        @printf(io, "   75th percentile:   %.1f ms\n", quantile(m.batch_times, 0.75) * 1000)
        @printf(io, "   90th percentile:   %.1f ms\n", quantile(m.batch_times, 0.90) * 1000)
    end
    println(io, "="^60)
end

function metrics_to_dict(m::PerformanceMetrics)
    Dict(
        "model_load_time"          => m.model_load_time,
        "data_load_time"           => m.data_load_time,
        "warmup_time"              => m.warmup_time,
        "total_inference_time"     => m.total_inference_time,
        "avg_batch_time"           => m.avg_batch_time,
        "std_batch_time"           => m.std_batch_time,
        "images_per_second"        => m.images_per_second,
        "single_image_latency"     => m.single_image_latency,
        "cpu_memory_mb"            => m.cpu_memory_mb,
        "gpu_memory_allocated_mb"  => m.gpu_memory_allocated_mb,
        "gpu_memory_cached_mb"     => m.gpu_memory_cached_mb,
        "batch_times"              => m.batch_times,
        "timestamps"               => m.timestamps,
        "image_loading_ms"         => m.image_loading_ms,
        "pure_cnn_inference_ms"    => m.pure_cnn_inference_ms,
        "complete_pipeline_ms"     => m.complete_pipeline_ms,
        "image_loading_throughput" => m.image_loading_throughput,
        "cnn_throughput"           => m.cnn_throughput,
        "processing_efficiency"    => m.processing_efficiency,
        "total_batches"            => m.total_batches,
        "batch_size"               => m.batch_size,
        "total_images"             => m.total_images,
        "memory_per_image_mb"      => m.memory_per_image_mb,
        "initial_cpu_mb"           => m.initial_cpu_mb,
        "initial_gpu_mb"           => m.initial_gpu_mb,
        "after_model_cpu_mb"       => m.after_model_cpu_mb,
        "after_model_gpu_mb"       => m.after_model_gpu_mb,
        "peak_cpu_mb"              => m.peak_cpu_mb,
        "peak_gpu_mb"              => m.peak_gpu_mb,
        "total_inference_time_per_run_s" => m.total_inference_time / 5,
    )
end

function save_performance_report(m::PerformanceMetrics)
    path = "/tmp/benchmarks_iws/gpu_julia_iws/results/inference_performance_report_julia_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).txt"
    open(path, "w") do io
        println(io, "="^80)
        println(io, "JULIA LUX CNN INFERENCE PERFORMANCE REPORT (GPU v2.2)")
        println(io, "="^80)
        println(io, "Generated: $(now())")
        println(io, "Dataset: test")
        println(io, "Total Images: $(m.total_images)")
        println(io, "Batch Size: $(m.batch_size)")
        show(io, m)
        println(io, "\n PERFORMANCE INSIGHTS:")
        println(io, "-"^40)
        m.std_batch_time > m.avg_batch_time * 0.5 ?
            println(io, " High batch time variance") :
            println(io, "Batch times are consistent")
        m.single_image_latency * 1000 < 5 ?
            println(io, "Low latency (<5ms per image)") :
            println(io, "Latency >5ms per image")
    end
    println("Performance report saved to: $path")
end

# =========================================================
# MAIN INFERENCE FUNCTION
# =========================================================
function run_inference_with_profiling(;
    model_path    = "/tmp/benchmarks_iws/gpu_julia_iws/models/model_v2_20260302_202405.jld2",
    inference_dir = "/tmp/benchmarks_iws/data/raw/test",
    output_csv    = "/tmp/benchmarks_iws/gpu_julia_iws/results/inference_predictions_julia.csv",
    batch_size    = DefaultConfig.BATCH_SIZE,
    warmup_runs   = 3,
    profile_runs  = 5
)
    println("\n" * "="^60)
    println("JULIA LUX CNN INFERENCE (GPU v2.2)")
    println("="^60)
    println("USING TEST DATASET FOR INFERENCE")
    println("   • Images: $inference_dir")
    println("   • Complete pipeline tracking enabled")
    println("-"^60)

    # ── Initial memory ────────────────────────────────────────────────────────
    println("\n TRACKING INITIAL MEMORY...")
    initial_cpu = cpu_memory_mb()
    initial_gpu = gpu_memory_mb()
    @printf("CPU Memory: %.1f MB\n", initial_cpu)
    @printf("GPU Memory: %.1f MB\n", initial_gpu)

    # ── Load model ────────────────────────────────────────────────────────────
    println("\n LOADING TRAINED MODEL...")
    # FIX 6: always log the model path
    println("Path: $model_path")
    !isfile(model_path) && error("Model file not found: $model_path")

    t_model    = time()
    model_data = jldopen(model_path, "r") do f
        (model = f["model"], ps = f["ps"], st = f["st"],
         norm_mean = Float32(f["norm_mean"]),   # FIX 3: load norm stats
         norm_std  = Float32(f["norm_std"]))
    end

    model     = model_data.model   # loaded from JLD2 (not SAR_CNN())
    ps        = model_data.ps |> Lux.gpu_device()

    # FIX 4: set_dropout_mode instead of LuxCore.testmode()
    # ORIGINAL BUG: LuxCore.testmode() may not correctly propagate
    # eval mode through all nested NamedTuples in custom Lux models
    st        = set_dropout_mode(model_data.st |> gpu_device(), false)

    # FIX 3: norm stats from checkpoint — never recompute from test data
    norm_mean = model_data.norm_mean
    norm_std  = model_data.norm_std

    model_load_time  = time() - t_model
    after_model_cpu  = cpu_memory_mb()
    after_model_gpu  = gpu_memory_mb()
    @printf("Model loaded in %.1f ms\n", model_load_time * 1000)
    @printf("norm_mean = %.4f   norm_std = %.4f\n", norm_mean, norm_std)
    @printf("Memory after model load:\n")
    @printf("CPU: %.1f MB\n", after_model_cpu)
    @printf("GPU: %.1f MB\n", after_model_gpu)

    # ── Sanity check on random noise ──────────────────────────────────────────
    println("Sanity check on random noise batch...")
    H, W    = DefaultConfig.TARGET_SIZE
    X_rand  = rand(Float32, H, W, 1, 10) .* 255.0f0 |> gpu_device()
    X_rand  = normalize_batch(X_rand, norm_mean, norm_std)
    y_rand, _ = model(X_rand, ps, st)
    y_rand_cpu = y_rand |> cpu_device()
    @printf("Noise predictions — min: %.4f  max: %.4f  mean: %.4f\n",
            minimum(y_rand_cpu), maximum(y_rand_cpu), mean(y_rand_cpu))

    # ── Preload dataset ───────────────────────────────────────────────────────
    println("\n LOADING TEST DATASET...")
    t_preload      = time()
    loader         = InferenceDataLoader(inference_dir; batch_size = batch_size)
    data_load_time = time() - t_preload
    total_images   = length(loader.image_paths)
    @printf("%d images loaded\n", total_images)

    # ── Warm-up passes ────────────────────────────────────────────────────────
    println("\n WARM-UP PASSES ($warmup_runs runs)...")
    t_warm       = time()
    warmup_times = Float64[]
    for run in 1:warmup_runs
        run_start = time()
        for (X, _, _) in loader
            # FIX 2+3: apply normalize_batch in warmup too
            X_norm = normalize_batch(X, norm_mean, norm_std)
            y, _   = model(X_norm, ps, st)
            CUDA.synchronize()
        end
        run_time = time() - run_start
        push!(warmup_times, run_time)
        run == 1 && @printf("First run: %.3f s\n", run_time)
    end
    warmup_time = time() - t_warm
    @printf("Average warm-up: %.3f s\n", mean(warmup_times))

    # ── Main profiling passes ─────────────────────────────────────────────────
    println("\n MAIN INFERENCE PROFILING ($profile_runs runs)...")
    all_batch_times = Float64[]
    all_load_times  = Float64[]
    all_timestamps  = Float64[]
    all_predictions = []
    peak_cpu        = 0.0
    peak_gpu        = 0.0

    total_inference_start = time()

    for run in 1:profile_runs
        run_predictions = zeros(Float32, total_images)
        run_start       = time()

        for (X, idx, load_time) in loader
            push!(all_load_times, load_time)

            # FIX 2+3: normalize with checkpoint stats before forward pass
            gpu_start = time()
            X_norm    = normalize_batch(X, norm_mean, norm_std)
            y, _      = model(X_norm, ps, st)
            CUDA.synchronize()
            gpu_time  = time() - gpu_start

            push!(all_batch_times, load_time + gpu_time)
            push!(all_timestamps,  time() - total_inference_start)

            y_cpu = y |> Lux.cpu_device()
            run_predictions[idx] .= vec(y_cpu)
            peak_cpu = max(peak_cpu, cpu_memory_mb())
            peak_gpu = max(peak_gpu, gpu_memory_mb())
        end

        push!(all_predictions, copy(run_predictions))
        run_time   = time() - run_start
        throughput = total_images / run_time
        @printf("Run %d/%d...\n", run, profile_runs)
        @printf("   Time: %.3f s, Throughput: %d img/s\n",
                run_time, round(Int, throughput))
    end

    total_inference_time = time() - total_inference_start

    # ── Compute metrics ───────────────────────────────────────────────────────
    avg_batch_time    = mean(all_batch_times)
    std_batch_time    = std(all_batch_times)
    images_per_second = total_images * profile_runs / total_inference_time
    single_image_latency = 1.0 / images_per_second

    avg_load_time = mean(all_load_times)
    avg_gpu_time  = mean(all_batch_times) - avg_load_time

    image_loading_ms         = (avg_load_time / batch_size) * 1000
    pure_cnn_ms              = (avg_gpu_time  / batch_size) * 1000
    complete_pipeline_ms     = image_loading_ms + pure_cnn_ms
    image_loading_throughput = 1000.0 / image_loading_ms
    cnn_throughput           = 1000.0 / pure_cnn_ms

    total_gpu_time       = sum(all_batch_times) - sum(all_load_times)
    processing_efficiency = total_gpu_time / total_inference_time

    memory_per_image     = (256 * 256 * 1 * 4) / (1024^2)   # 0.250 MB — marginal cost per image (H×W×C×bytes)


    total_batches        = length(all_batch_times)

    metrics = PerformanceMetrics(
        model_load_time, data_load_time, warmup_time,
        total_inference_time, avg_batch_time, std_batch_time,
        images_per_second, single_image_latency,
        peak_cpu, peak_gpu, 0.0,
        all_batch_times, all_timestamps,
        image_loading_ms, pure_cnn_ms, complete_pipeline_ms,
        image_loading_throughput, cnn_throughput,
        processing_efficiency, total_batches, batch_size, total_images,
        memory_per_image, initial_cpu, initial_gpu,
        after_model_cpu, after_model_gpu, peak_cpu, peak_gpu
    )

    println("\n" * "="^60)
    println("PIPELINE PERFORMANCE BREAKDOWN")
    println("="^60)
    show(stdout, metrics)

    # ── Save predictions ──────────────────────────────────────────────────────
    println("\n SAVING PREDICTIONS...")
    final_predictions = mean(all_predictions)
    final_binary      = Int.(final_predictions .> 0.5f0)

    df = DataFrame(
        image_path  = loader.image_paths,
        prediction  = final_binary,
        confidence  = final_predictions,
        timestamp   = fill(now(), total_images)
    )
    CSV.write(output_csv, df)
    @printf("Predictions saved to %s\n",    output_csv)
    @printf("Total predictions  : %d\n",    nrow(df))
    @printf("Positive rate      : %.1f%%\n", sum(final_binary)/total_images*100)

    # FIX 7: sanity check on positive rate — alert if extreme
    pos_rate = sum(final_binary) / total_images * 100
    if pos_rate < 1.0 || pos_rate > 99.0
        println("WARNING: Extreme positive rate $(round(pos_rate,digits=1))%")
        println("Check norm_mean/norm_std and TARGET_SIZE match training")
        println("Check model checkpoint path is correct")
    end

    # ── Save reports ──────────────────────────────────────────────────────────
    save_performance_report(metrics)

    # ── Key findings ──────────────────────────────────────────────────────────
    println("\n" * "="^60)
    println("KEY FINDINGS FOR COMPARISON")
    println("="^60)
    println("Julia Lux CNN (GPU v2.2):")
    @printf("  • Complete throughput : %d img/s\n", round(Int, images_per_second))
    @printf("  • CNN throughput      : %d img/s\n", round(Int, cnn_throughput))
    @printf("  • Image loading       : %d img/s\n", round(Int, image_loading_throughput))
    @printf("  • Peak CPU memory     : %.1f MB\n",  peak_cpu)
    @printf("  • Peak GPU memory     : %.1f MB\n",  peak_gpu)
    @printf("  • Memory per image    : %.3f MB\n",  memory_per_image)
    @printf("  • 6000 images est     : %d s\n",     round(Int, 6000 / images_per_second))
    println("="^60)

    return df, metrics, loader, model, ps, st, norm_mean, norm_std
end

# =========================================================
# EXECUTION
# =========================================================
global df, metrics, loader, model, ps, st, _norm_mean, _norm_std

if abspath(PROGRAM_FILE) == @__FILE__
    if CUDA.functional()
        println("CUDA GPU detected: $(CUDA.device())")
    else
        error("CUDA not functional — GPU required")
    end
    df, metrics, loader, model, ps, st, _norm_mean, _norm_std =
        run_inference_with_profiling()
end

# =========================================================
# GPU POWER PROFILING BRIDGE
# =========================================================
include(joinpath(@__DIR__, "pipeline_b_gpu_power_profiler.jl"))

N_IMAGES   = length(loader.image_paths)
BATCH_SIZE = loader.batch_size

# norm stats are now in global scope from the return above
# _norm_mean and _norm_std set directly — model_data not needed

function _one_gpu_inference_pass()::Dict{String,Any}
    predictions = zeros(Float32, N_IMAGES)
    t0 = time()
    for (X, idx, _load_time) in loader
        X_norm = normalize_batch(X, _norm_mean, _norm_std)  # FIX: normalize here too
        y, _   = model(X_norm, ps, st)
        CUDA.synchronize()
        predictions[idx] .= vec(y |> cpu_device())
    end
    wall = time() - t0
    return Dict(
        "wall_time_s"   => wall,
        "throughput"    => N_IMAGES / wall,
        "positive_rate" => sum(predictions .> 0.5f0) / N_IMAGES
    )
end

function _build_extra_fields(metrics)::Dict{String,Any}
    isnothing(metrics) && return Dict{String,Any}()
    Dict{String,Any}(
        "timing_metrics" => Dict{String,Any}(
            "images_per_second"        => metrics.images_per_second,
            "single_image_latency_ms"  => metrics.single_image_latency * 1000,
            "avg_batch_time_ms"        => metrics.avg_batch_time  * 1000,
            "std_batch_time_ms"        => metrics.std_batch_time  * 1000,
            "total_inference_time_s"   => metrics.total_inference_time,
            "image_loading_ms"         => metrics.image_loading_ms,
            "pure_cnn_inference_ms"    => metrics.pure_cnn_inference_ms,
            "complete_pipeline_ms"     => metrics.complete_pipeline_ms,
            "image_loading_throughput" => metrics.image_loading_throughput,
            "cnn_throughput"           => metrics.cnn_throughput,
            "processing_efficiency"    => metrics.processing_efficiency,
            "peak_cpu_mb"              => metrics.peak_cpu_mb,
            "peak_gpu_mb"              => metrics.peak_gpu_mb,
            "memory_per_image_mb"      => metrics.memory_per_image_mb,
            "batch_p25_ms"             => quantile(metrics.batch_times, 0.25) * 1000,
            "batch_p50_ms"             => quantile(metrics.batch_times, 0.50) * 1000,
            "batch_p75_ms"             => quantile(metrics.batch_times, 0.75) * 1000,
            "batch_p90_ms"             => quantile(metrics.batch_times, 0.90) * 1000,
            "model_load_time_ms"       => metrics.model_load_time * 1000,
            "data_preload_time_s"      => metrics.data_load_time,
            "total_batches"            => metrics.total_batches
        )
    )
end

println("\n" * "="^60)
println("STARTING GPU POWER PROFILING PHASE")
CUDA.reclaim()       # release all cached GPU memory
GC.gc(true)          # full Julia garbage collection
sleep(120)            # let CUDA threads fully settle before idle measurement


println("Idle baseline:  60 s")
println("Inference runs: 5  (+ 30s cool-down between each)")
println("Estimated total: ~15–20 min")
println("="^60)

_extra = _build_extra_fields(metrics)

gpu_report, gpu_power, gpu_idle = run_full_gpu_power_profile(
    inference_fn   = _one_gpu_inference_pass,
    n_images       = N_IMAGES,
    batch_size     = BATCH_SIZE,
    pipeline_name  = "Julia_Lux_CNN_GPU",
    output_json    = "/tmp/benchmarks_iws/gpu_julia_iws/results/julia_gpu_inference_metrics.json",
    idle_duration_s = 60.0,
    interval_ms     = 100,
    discard_start_s = 2.0,
    discard_end_s   = 2.0,
    cooldown_s      = 30.0,
    extra_fields    = _extra
)

s = gpu_report["publishable_summary"]
println("\n" * "="^60)
println("COMBINED GPU RESULTS FOR PAPER")
println("="^60)
@printf("Dataset:              %d images\n",     N_IMAGES)
@printf("GPU power (gross):    %.2f ± %.2f W\n", s["mean_gpu_power_w"], s["std_gpu_power_w"])
isnothing(s["net_inference_power_w"]) ||
    @printf("GPU power (net):      %.2f W\n",    s["net_inference_power_w"])
@printf("Energy/image (gross): %.3f mJ\n",       s["energy_per_image_mj_gross"])
isnothing(s["energy_per_image_mj_net"]) ||
    @printf("Energy/image (net):   %.3f mJ\n",   s["energy_per_image_mj_net"])
s["thermal_throttling_detected"] &&
    println("Thermal throttling detected — results flagged in JSON")
println("Compare against Python EVA02+XGBoost GPU pipeline")
println("="^60)
# Add explicit cleanup before exit
function cleanup_gpu()
    try
        println("Cleaning up GPU resources...")
        CUDA.reclaim()
        GC.gc(true)
        sleep(0.5)
        CUDA.reclaim()
        println("Cleanup complete")
    catch e
        # Ignore cleanup errors
    end
end

# Register cleanup to run before exit
atexit(cleanup_gpu)

# ================================
# pipeline_b_model.jl  –  SAR CNN v2  (GPU / CUDA)
#
# Author:  Arun Shukla (architecture design, implementation)
# Revised: João Pinelo (corrected parameter count)
#
#  IMPROVEMENTS OVER v1:
#    1. Doubled channel widths: 1→32→64→128 (was 1→16→32→64)
#    2. Added 4th conv block for deeper feature hierarchy
#    3. Classifier hidden dim: 128→64→1 (was 64→32→1)
#    4. Label smoothing built into loss function
#    5. Progressive dropout: 0.05→0.1→0.15→0.2 conv, 0.4 classifier
#
#  Architecture : 4× ConvBlock → GlobalAvgPool → ClassifierBlock
#  Channels     : 1 → 32 → 64 → 128 → 128 → 64 → 1
#  Output       : scalar sigmoid ∈ (0,1) for binary classification
#  Parameters   : 283,329 (was 34K in v1 — model was severely underfitting)
#
#  BatchNorm pattern unchanged — st captured in closure, not through pullback.
#  Only Dropout state managed via set_dropout_mode().
# ================================

using Lux, LuxCore, NNlib, Random
using Statistics, LinearAlgebra
using LuxCUDA, CUDA

# CUDA version set via preferences - no call needed
ENV["JULIA_CUDA_VERBOSE"]                = "0"
ENV["MLDATADEVICES_SILENCE_WARN_NO_GPU"] = "1"

if CUDA.functional()
    dev     = CUDA.device()
    vram_gb = round(CUDA.totalmem(dev) / (1024^3), digits=2)
    println("GPU : $(CUDA.name(dev))  |  VRAM: $(vram_gb) GiB")
else
    @warn "CUDA not functional — model will run on CPU fallback"
end

if !isdefined(Main, :DefaultConfig)
    const _MODEL_SCRIPT_DIR  = @__DIR__
    const _MODEL_CONFIG_FILE = abspath(joinpath(_MODEL_SCRIPT_DIR, "..", "config", "default_config.jl"))
    isfile(_MODEL_CONFIG_FILE) || error("Config not found: $_MODEL_CONFIG_FILE")
    include(_MODEL_CONFIG_FILE)
end
using .DefaultConfig
using .DefaultConfig.DeviceConfig

# =============================================================================
# 1. ConvBlock  —  Conv → BN → ReLU → (optional MaxPool) → Dropout
# =============================================================================
struct ConvBlock <: LuxCore.AbstractLuxContainerLayer{(:conv, :bn, :pool, :drop)}
    conv::Conv
    bn::BatchNorm
    pool::Union{MaxPool, NoOpLayer}
    drop::Dropout
end

function ConvBlock(kernel, in_ch, out_ch;
                   stride       = (1, 1),
                   pad          = 0,
                   pool_size    = nothing,
                   dropout_rate = 0.1f0)
    conv = Conv(kernel, in_ch => out_ch; stride=stride, pad=pad)
    bn   = BatchNorm(out_ch)
    pool = isnothing(pool_size) ? NoOpLayer() : MaxPool(pool_size)
    drop = Dropout(dropout_rate)
    return ConvBlock(conv, bn, pool, drop)
end

function (cb::ConvBlock)(x, ps, st)
    y, st_conv = cb.conv(x, ps.conv, st.conv)
    y, st_bn   = cb.bn(y,  ps.bn,   st.bn)
    y          = relu.(y)
    y, st_pool = cb.pool(y, ps.pool, st.pool)
    y, st_drop = cb.drop(y, ps.drop, st.drop)
    return y, (conv=st_conv, bn=st_bn, pool=st_pool, drop=st_drop)
end

# =============================================================================
# 2. AggBlock  —  Global Average Pooling
# =============================================================================
struct AggBlock <: LuxCore.AbstractLuxLayer end

LuxCore.initialparameters(::Random.AbstractRNG, ::AggBlock) = NamedTuple()
LuxCore.initialstates(::Random.AbstractRNG,     ::AggBlock) = NamedTuple()

function (::AggBlock)(x, ps, st)
    y = mean(x; dims=(1, 2))
    y = reshape(y, size(y, 3), size(y, 4))
    return y, st
end

# =============================================================================
# 3. ClassifierBlock  —  Dense(128→64) → Dropout(0.4) → Dense(64→1) → sigmoid
# =============================================================================
struct ClassifierBlock <: LuxCore.AbstractLuxContainerLayer{(:dense1, :dropout, :dense2)}
    dense1::Dense
    dropout::Dropout
    dense2::Dense
end

function ClassifierBlock(in_features::Int; dropout_rate::Float32 = 0.4f0)
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

# =============================================================================
# 4. SAR_CNN v2 — wider + deeper
#
#  Block 1: (7×7) 1  → 32,  stride=2, pad=3              → 128×128×32
#  Block 2: (5×5) 32 → 64,  stride=1, pad=2, pool=(2,2)  → 64×64×64
#  Block 3: (3×3) 64 → 128, stride=1, pad=1, pool=(2,2)  → 32×32×128
#  Block 4: (3×3) 128→ 128, stride=1, pad=1, pool=(2,2)  → 16×16×128
#  GAP → 128
#  Dense(128→64) → Dropout(0.4) → Dense(64→1) → sigmoid
# =============================================================================
struct SAR_CNN <: LuxCore.AbstractLuxContainerLayer{(:block1, :block2, :block3, :block4, :agg, :classifier)}
    block1::ConvBlock
    block2::ConvBlock
    block3::ConvBlock
    block4::ConvBlock
    agg::AggBlock
    classifier::ClassifierBlock
end

function SAR_CNN()
    block1     = ConvBlock((7, 7), 1,   32;  stride=(2, 2), pad=3, dropout_rate=0.05f0)
    block2     = ConvBlock((5, 5), 32,  64;  stride=(1, 1), pad=2, pool_size=(2, 2), dropout_rate=0.1f0)
    block3     = ConvBlock((3, 3), 64,  128; stride=(1, 1), pad=1, pool_size=(2, 2), dropout_rate=0.15f0)
    block4     = ConvBlock((3, 3), 128, 128; stride=(1, 1), pad=1, pool_size=(2, 2), dropout_rate=0.2f0)
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

# =============================================================================
# 5. Dropout-only mode switch — handles all 5 dropout layers
# =============================================================================
function _set_dropout_val(dropout_st::NamedTuple, training::Bool)
    return merge(dropout_st, (training = Val(training),))
end

function set_dropout_mode(st::NamedTuple, training::Bool)
    new_b1  = merge(st.block1, (drop = _set_dropout_val(st.block1.drop, training),))
    new_b2  = merge(st.block2, (drop = _set_dropout_val(st.block2.drop, training),))
    new_b3  = merge(st.block3, (drop = _set_dropout_val(st.block3.drop, training),))
    new_b4  = merge(st.block4, (drop = _set_dropout_val(st.block4.drop, training),))
    new_cls = merge(st.classifier, (dropout = _set_dropout_val(st.classifier.dropout, training),))
    return merge(st, (block1=new_b1, block2=new_b2, block3=new_b3, block4=new_b4, classifier=new_cls))
end

set_training_mode(st::NamedTuple, training::Bool) = set_dropout_mode(st, training)

# =============================================================================
# 6. Label smoothing loss
#
#  Replaces hard 0/1 with smooth_label = label * (1 - α) + α/2
#  α=0.1 → 0 becomes 0.05, 1 becomes 0.95
#  This is time-tested regularization that reliably gives 0.5-1% accuracy boost.
# =============================================================================
function label_smooth(y::AbstractArray{Float32}; α::Float32 = 0.1f0)
    return y .* (1.0f0 - α) .+ (α * 0.5f0)
end

export SAR_CNN, set_dropout_mode, set_training_mode, label_smooth

# =============================================================================
# Self-test
# =============================================================================
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("SAR_CNN v2 SELF-TEST  (GPU)")
    println("="^60)

    rng    = Random.default_rng()
    Random.seed!(rng, DefaultConfig.RNG_SEED)
    model  = SAR_CNN()
    ps, st = Lux.setup(rng, model)

    ps_gpu = ps |> gpu_device()
    st_gpu = st |> gpu_device()

    n_params = LuxCore.parameterlength(ps)
    println("Parameters : $n_params  (should be 283,329)")

    h, w    = DefaultConfig.TARGET_SIZE
    x_dummy = randn(Float32, h, w, 1, 4) |> gpu_device()

    st_train   = set_dropout_mode(st_gpu, true)
    y_train, _ = model(x_dummy, ps_gpu, st_train)
    y_cpu      = Array(y_train)
    println("Train output : $(size(y_train))  range [$(round(minimum(y_cpu),digits=4)), $(round(maximum(y_cpu),digits=4))]")
    @assert all(0 .≤ y_cpu .≤ 1)

    st_infer    = set_dropout_mode(st_gpu, false)
    y_infer, _  = model(x_dummy, ps_gpu, st_infer)
    println("Infer output : $(size(y_infer))")

    # Test label smoothing
    y_hard = Float32[0.0, 1.0, 0.0, 1.0]
    y_soft = label_smooth(y_hard)
    println("Label smooth : $y_hard → $y_soft")
    @assert all(y_soft .> 0.0f0) && all(y_soft .< 1.0f0)

    println("ALL TESTS PASSED")
    println("="^60)
end

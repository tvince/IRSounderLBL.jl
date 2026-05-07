"""
Compute backend auto-detection and strategy selection.

Supports:
- Apple Metal GPU (development on MacBook)
- NVIDIA CUDA GPU (production)
- CPU with multi-threading (fallback)
"""

"""
    ComputeBackend

Describes the selected compute backend for GPU-agnostic kernels.

# Fields
- `use_gpu`:     whether a GPU is available and selected
- `gpu_backend`: symbol `:metal`, `:cuda`, or `:none`
- `n_threads`:   number of CPU threads available
- `n_gpu_cores`: approximate number of GPU cores (0 if no GPU)
"""
struct ComputeBackend
    use_gpu::Bool
    gpu_backend::Symbol
    n_threads::Int
    n_gpu_cores::Int
end

"""
    detect_backend(; prefer_gpu=true, verbose=true) -> ComputeBackend

Auto-detect the best available compute backend.

Checks for Metal (Apple GPU) first, then CUDA (NVIDIA GPU), then falls
back to CPU threads.  Set `prefer_gpu=false` to force CPU.
"""
function detect_backend(; prefer_gpu::Bool = true, verbose::Bool = true)
    n_threads = Threads.nthreads()

    if prefer_gpu
        # Try Metal (Apple GPU)
        metal_available = _check_metal()
        if metal_available
            verbose && @info "Backend: Apple Metal GPU detected"
            return ComputeBackend(true, :metal, n_threads, 0)
        end

        # Try CUDA (NVIDIA GPU)
        cuda_available = _check_cuda()
        if cuda_available
            n_cores = _cuda_cores()
            verbose && @info "Backend: NVIDIA CUDA GPU detected ($n_cores cores)"
            return ComputeBackend(true, :cuda, n_threads, n_cores)
        end
    end

    verbose && @info "Backend: CPU with $n_threads thread(s)"
    return ComputeBackend(false, :none, n_threads, 0)
end

"""
    ka_backend(backend::ComputeBackend)

Return the KernelAbstractions backend object corresponding to a `ComputeBackend`.
"""
function ka_backend(cb::ComputeBackend)
    if cb.gpu_backend == :metal
        return _metal_backend()
    elseif cb.gpu_backend == :cuda
        return _cuda_backend()
    else
        return CPU()
    end
end

# ── Internal helpers ─────────────────────────────────────────────────────────

function _check_metal()
    try
        # Metal.jl is an extension dep; check without importing at load time
        return isdefined(Main, :Metal) && Main.Metal.functional()
    catch
        return false
    end
end

function _check_cuda()
    try
        return isdefined(Main, :CUDA) && Main.CUDA.functional()
    catch
        return false
    end
end

function _cuda_cores()
    try
        dev = Main.CUDA.device()
        return Main.CUDA.attribute(dev, Main.CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * 128
    catch
        return 0
    end
end

function _metal_backend()
    try
        return Main.Metal.MetalBackend()
    catch
        return CPU()
    end
end

function _cuda_backend()
    try
        return Main.CUDA.CUDABackend()
    catch
        return CPU()
    end
end

Base.show(io::IO, cb::ComputeBackend) =
    print(io, "ComputeBackend(gpu=$(cb.use_gpu), backend=:$(cb.gpu_backend), " *
              "threads=$(cb.n_threads))")

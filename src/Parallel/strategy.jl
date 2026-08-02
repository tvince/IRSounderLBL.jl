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

# ── GPU extension hooks ──────────────────────────────────────────────────────
# Metal and CUDA are optional weak dependencies. Loading either (`using Metal` /
# `using CUDA`) activates a package extension that overrides these hooks with the
# real implementations. Without the extension the defaults report "no GPU", so the
# package works with GPU deps absent. `Val(:metal)` / `Val(:cuda)` select backend.
_gpu_functional(::Val)  = false
_gpu_cores(::Val)       = 0
_gpu_ka_backend(::Val)  = CPU()

# ── Internal helpers ─────────────────────────────────────────────────────────

_check_metal()   = _gpu_functional(Val(:metal))
_check_cuda()    = _gpu_functional(Val(:cuda))
_cuda_cores()    = _gpu_cores(Val(:cuda))
_metal_backend() = _gpu_ka_backend(Val(:metal))
_cuda_backend()  = _gpu_ka_backend(Val(:cuda))

Base.show(io::IO, cb::ComputeBackend) =
    print(io, "ComputeBackend(gpu=$(cb.use_gpu), backend=:$(cb.gpu_backend), " *
              "threads=$(cb.n_threads))")

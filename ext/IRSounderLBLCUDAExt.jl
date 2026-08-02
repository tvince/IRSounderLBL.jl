module IRSounderLBLCUDAExt

# Activated automatically when the user loads `CUDA` alongside IRSounderLBL.
# Provides the NVIDIA-GPU implementations of the backend hooks declared in
# src/Parallel/strategy.jl.
using CUDA
import IRSounderLBL: _gpu_functional, _gpu_cores, _gpu_ka_backend

_gpu_functional(::Val{:cuda}) = CUDA.functional()
_gpu_ka_backend(::Val{:cuda}) = CUDA.CUDABackend()

function _gpu_cores(::Val{:cuda})
    dev = CUDA.device()
    return CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) * 128
end

end # module

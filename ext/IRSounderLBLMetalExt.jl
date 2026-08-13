module IRSounderLBLMetalExt

# Activated automatically when the user loads `Metal` alongside IRSounderLBL.
# Provides the Apple-GPU implementations of the backend hooks declared in
# src/Parallel/strategy.jl, plus the Float32 precision override for voigt.jl.
using Metal
import IRSounderLBL: _gpu_functional, _gpu_ka_backend, _backend_float_type

_gpu_functional(::Val{:metal}) = Metal.functional()
_gpu_ka_backend(::Val{:metal}) = Metal.MetalBackend()

# Apple Metal does not support Float64.
_backend_float_type(::Metal.MetalBackend) = Float32

end # module

using Interpolations

"""
    interp_profile(p_ref, x_ref, p_target)

Linearly interpolate a profile `x_ref` defined on pressure levels `p_ref`
onto target pressure levels `p_target`.  Interpolation is linear in log(p)
which is the natural coordinate for atmospheric profiles.
"""
function interp_profile(p_ref::AbstractVector{<:Real},
                        x_ref::AbstractVector{<:Real},
                        p_target::AbstractVector{<:Real})
    logp_ref = log.(p_ref)
    # linear_interpolation requires strictly increasing knots
    if !issorted(logp_ref)
        idx = sortperm(logp_ref)
        logp_ref = logp_ref[idx]
        x_ref    = x_ref[idx]
    end
    itp = linear_interpolation(logp_ref, x_ref; extrapolation_bc=Flat())
    return itp.(log.(p_target))
end

"""
    interp_vmr(p_ref, vmr_ref, p_target)

Interpolate a VMR profile in log(VMR)–log(p) space to better preserve
the orders-of-magnitude variation typical of trace gas profiles.
"""
function interp_vmr(p_ref::AbstractVector{<:Real},
                    vmr_ref::AbstractVector{<:Real},
                    p_target::AbstractVector{<:Real})
    logp_ref = log.(p_ref)
    logvmr   = log.(max.(vmr_ref, 1e-20))
    if !issorted(logp_ref)
        idx      = sortperm(logp_ref)
        logp_ref = logp_ref[idx]
        logvmr   = logvmr[idx]
    end
    itp = linear_interpolation(logp_ref, logvmr; extrapolation_bc=Flat())
    return exp.(itp.(log.(p_target)))
end

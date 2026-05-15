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

Interpolate a VMR profile linearly in log(p) space.

Uses the same scheme as `interp_profile` (linear VMR in log-p), which is
consistent with ARTS's trapezoidal integration at pressure-level boundaries.
Log-VMR interpolation was previously used here but systematically
underestimates H2O column amounts by 1–8% in layers where H2O drops rapidly,
causing a +0.4 K warm bias in the H2O 6 µm band compared to ARTS.
"""
function interp_vmr(p_ref::AbstractVector{<:Real},
                    vmr_ref::AbstractVector{<:Real},
                    p_target::AbstractVector{<:Real})
    return interp_profile(p_ref, vmr_ref, p_target)
end

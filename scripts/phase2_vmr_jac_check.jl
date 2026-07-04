# Phase 2 VMR Jacobian — measure analytic vs finite-difference agreement.
using IRSounderLBL
using Printf

# ── (a) Isolate the gradient helper: _cg_column_vmr_grad vs scalar central FD ──
println("=== _cg_column_vmr_grad vs scalar FD ===")
let
    cases = [(4.0e-4, 4.0e-4, 1000.0, 600.0),   # well-mixed
             (4.0e-4, 4.0e-4, 600.0, 200.0),
             (1.0e-2, 3.0e-3, 1000.0, 600.0),   # strong gradient (H2O-like)
             (3.0e-3, 1.0e-2, 600.0, 200.0),    # increasing with height
             (5.0e-4, 4.9e-4, 850.0, 700.0)]    # nearly equal
    for (v1, v2, p1, p2) in cases
        g1, g2 = IRSounderLBL._cg_column_vmr_grad(v1, v2, p1, p2)
        h1 = v1 * 1e-6; h2 = v2 * 1e-6
        f1 = (IRSounderLBL.cg_column_vmr(v1+h1, v2, p1, p2) -
              IRSounderLBL.cg_column_vmr(v1-h1, v2, p1, p2)) / (2h1)
        f2 = (IRSounderLBL.cg_column_vmr(v1, v2+h2, p1, p2) -
              IRSounderLBL.cg_column_vmr(v1, v2-h2, p1, p2)) / (2h2)
        @printf("  v=(%.2e,%.2e) p=(%.0f,%.0f): ∂v1 an=%.6e fd=%.6e | ∂v2 an=%.6e fd=%.6e\n",
                v1, v2, p1, p2, g1, f1, g2, f2)
    end
end

mk(ν0, S) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                       Float32(0.08), Float32(0.10), 250.0,
                       Float32(0.75), Float32(0.0))
ll = HITRANLinelist([mk(700.5, 3.0e-23), mk(702.0, 1.0e-23)])
linelists = Dict(CO2 => ll)
iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)

function check(prof; tag="")
    spec = StateVectorSpec(length(prof.temperature), [CO2];
                           include_temperature=false,
                           include_tsfc=true, include_emissivity=true)
    fm = (iasi=iasi, apply_continuum=false, with_ils=false)
    jac_fd = finite_difference_jacobian(prof, linelists, spec;
                                        T_sfc=290.0, ε_sfc=0.98,
                                        observable=:bt, fm_kwargs=fm)
    jac_an = analytic_jacobian(prof, linelists, spec;
                               iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
                               observable=:bt, apply_continuum=false, with_ils=false)
    co2 = spec.vmr_ranges[1][2]
    a = jac_an.K[:, co2]; f = jac_fd.K[:, co2]
    @printf("%-22s n_lev=%2d  VMR maxabsΔ=%.3e  rel=%.3e\n",
            tag, length(prof.temperature),
            maximum(abs.(a .- f)), maximum(abs.(a .- f))/(maximum(abs.(f))+1e-30))
end

# ── (b) Does the VMR residual collapse as layers thin? ──
println("\n=== analytic vs FD (VMR block), continuum & ILS off ===")
# coarse 2-layer (extreme)
check(AtmosphericProfile([1000.0, 600.0, 200.0], [288.0, 255.0, 225.0],
                         [0.0, 4.0, 12.0], Dict(CO2 => fill(4.0e-4, 3))); tag="coarse 2-layer")
# isothermal 2-layer (kills T_cg coupling)
check(AtmosphericProfile([1000.0, 600.0, 200.0], [260.0, 260.0, 260.0],
                         [0.0, 4.0, 12.0], Dict(CO2 => fill(4.0e-4, 3))); tag="isothermal 2-layer")
# finer layering over the same column
for nlev in (5, 11, 21)
    p = collect(range(1000.0, 200.0; length=nlev))
    T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
    z = (1000.0 .- p) ./ 66.0
    check(AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev))); tag="linear-T grid")
end

println("\n=== with_ils=true convergence (test config) ===")
function check_ils(prof; tag="")
    spec = StateVectorSpec(length(prof.temperature), [CO2];
                           include_temperature=false, include_tsfc=true, include_emissivity=true)
    fm = (iasi=iasi, apply_continuum=false, with_ils=true)
    fd = finite_difference_jacobian(prof, linelists, spec; T_sfc=290.0, ε_sfc=0.98,
                                    observable=:bt, fm_kwargs=fm)
    an = analytic_jacobian(prof, linelists, spec; iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
                           observable=:bt, apply_continuum=false, with_ils=true)
    co2 = spec.vmr_ranges[1][2]
    a = an.K[:, co2]; f = fd.K[:, co2]
    ts = maximum(abs.(an.K[:, spec.tsfc_index] .- fd.K[:, spec.tsfc_index]))
    ep = maximum(abs.(an.K[:, spec.emis_index] .- fd.K[:, spec.emis_index]))
    @printf("%-14s n_lev=%2d VMRmaxΔ=%.3e VMRrel=%.3e Tsfcabs=%.2e εabs=%.2e\n",
            tag, length(prof.temperature), maximum(abs.(a .- f)),
            maximum(abs.(a .- f))/(maximum(abs.(f))+1e-30), ts, ep)
end
for nlev in (6, 11)
    p = collect(range(1000.0, 200.0; length=nlev))
    T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
    z = (1000.0 .- p) ./ 66.0
    check_ils(AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev))); tag="ils grid")
end

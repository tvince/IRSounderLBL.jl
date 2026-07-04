# Phase 2b — does the VMR coupling close the residual vs FD?
using IRSounderLBL
using Printf

iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)

function vmr_rel(prof, linelists, sp, T_sfc; nlev)
    spec = StateVectorSpec(nlev, [sp]; include_temperature=false,
                           include_tsfc=true, include_emissivity=true)
    fm = (iasi=iasi, apply_continuum=false, with_ils=true)
    fd = finite_difference_jacobian(prof, linelists, spec; T_sfc=T_sfc, ε_sfc=0.98,
                                    observable=:bt, fm_kwargs=fm)
    co2 = spec.vmr_ranges[1][2]
    rel(an) = maximum(abs.(an.K[:, co2] .- fd.K[:, co2])) /
              (maximum(abs.(fd.K[:, co2])) + 1e-30)
    an_off = analytic_jacobian(prof, linelists, spec; iasi=iasi, T_sfc=T_sfc, ε_sfc=0.98,
                               observable=:bt, apply_continuum=false, with_ils=true,
                               vmr_coupling=false)
    an_on  = analytic_jacobian(prof, linelists, spec; iasi=iasi, T_sfc=T_sfc, ε_sfc=0.98,
                               observable=:bt, apply_continuum=false, with_ils=true,
                               vmr_coupling=true)
    return rel(an_off), rel(an_on)
end

# CO2: dominant residual is the p_cg (width) coupling.
mkC(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                           Float32(0.07), Float32(0.08), E, Float32(0.75), Float32(-0.006))
llC = Dict(CO2 => HITRANLinelist([mkC(700.5, 2.0e-21, 250.0), mkC(702.0, 8.0e-22, 600.0)]))
for nlev in (6, 11)
    p = collect(range(1000.0, 200.0; length=nlev))
    T = 288.0 .+ (225.0-288.0).*(1000.0.-p)./800.0
    z = (1000.0.-p)./66.0
    prof = AtmosphericProfile(p, T, z, Dict(CO2=>fill(4.0e-4,nlev)))
    off, on = vmr_rel(prof, llC, CO2, 290.0; nlev=nlev)
    @printf("CO2  nlev=%2d  coupling OFF rel=%.3e  ON rel=%.3e\n", nlev, off, on)
end

# H2O: exercises self-broadening coupling too (vmr_self = vmr_cg).
mkW(ν0, S, E) = HITRANLine(Int8(1), Int8(1), ν0, S, 1.0,
                           Float32(0.09), Float32(0.45), E, Float32(0.70), Float32(0.005))
llW = Dict(H2O => HITRANLinelist([mkW(700.6, 3.0e-21, 200.0), mkW(702.3, 1.2e-21, 450.0)]))
for nlev in (6, 11)
    p = collect(range(1000.0, 200.0; length=nlev))
    T = 288.0 .+ (225.0-288.0).*(1000.0.-p)./800.0
    z = (1000.0.-p)./66.0
    # decreasing H2O with height (realistic gradient → nontrivial CG p/T/self coupling)
    h2o = [0.01 * exp(-3.0*(1000.0-pk)/800.0) for pk in p]
    prof = AtmosphericProfile(p, T, z, Dict(H2O=>h2o))
    off, on = vmr_rel(prof, llW, H2O, 290.0; nlev=nlev)
    @printf("H2O  nlev=%2d  coupling OFF rel=%.3e  ON rel=%.3e\n", nlev, off, on)
end

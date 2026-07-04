# Phase 3 — analytic temperature Jacobian vs FD harness.
using IRSounderLBL
using Printf

mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                          Float32(0.07), Float32(0.08), E,
                          Float32(0.75), Float32(-0.003))
ll = HITRANLinelist([mk(700.5, 2.0e-21, 250.0), mk(702.0, 8.0e-22, 600.0)])
linelists = Dict(CO2 => ll)
iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)

function build(nlev)
    p = collect(range(1000.0, 200.0; length=nlev))
    T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
    z = (1000.0 .- p) ./ 66.0
    AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev)))
end

for (nlev, δT) in ((6, 0.5), (6, 0.1), (11, 0.1))
    prof = build(nlev)
    spec = StateVectorSpec(nlev, [CO2]; include_temperature=true,
                           include_tsfc=true, include_emissivity=true)
    fm = (iasi=iasi, apply_continuum=false, with_ils=true)
    steps = default_fd_steps(spec; δT=δT, δlogvmr=1e-3, δtsfc=δT, δε=1e-3)
    fd = finite_difference_jacobian(prof, linelists, spec; T_sfc=290.0, ε_sfc=0.98,
                                    observable=:bt, steps=steps, fm_kwargs=fm)
    an = analytic_jacobian(prof, linelists, spec; iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
                           observable=:bt, apply_continuum=false, with_ils=true)

    tr  = spec.temp_range
    co2 = spec.vmr_ranges[1][2]
    tblk = maximum(abs.(an.K[:, tr]  .- fd.K[:, tr]))
    trel = tblk / (maximum(abs.(fd.K[:, tr])) + 1e-30)
    vblk = maximum(abs.(an.K[:, co2] .- fd.K[:, co2]))
    ts   = maximum(abs.(an.K[:, spec.tsfc_index] .- fd.K[:, spec.tsfc_index]))
    ep   = maximum(abs.(an.K[:, spec.emis_index] .- fd.K[:, spec.emis_index]))
    @printf("nlev=%2d δT=%.2f  Tblk max|Δ|=%.3e rel=%.3e | VMRblk=%.3e Tsfc=%.2e ε=%.2e | y0=%.1e\n",
            nlev, δT, tblk, trel, vblk, ts, ep, maximum(abs.(an.y0 .- fd.y0)))
end

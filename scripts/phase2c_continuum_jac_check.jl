# Phase 2c — with continuum ON, do the H2O/CO2 VMR & T columns now match FD?
using IRSounderLBL
using Printf

iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)
nlev = 6
p = collect(range(1000.0, 200.0; length=nlev))
T = 288.0 .+ (225.0-288.0).*(1000.0.-p)./800.0
z = (1000.0.-p)./66.0

mkW(ν0, S, E) = HITRANLine(Int8(1), Int8(1), ν0, S, 1.0,
                           Float32(0.09), Float32(0.45), E, Float32(0.70), Float32(0.005))
mkC(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                           Float32(0.07), Float32(0.08), E, Float32(0.75), Float32(-0.006))

function run(sp, ll, profvmr, tag)
    prof = AtmosphericProfile(p, T, z, profvmr)
    spec = StateVectorSpec(nlev, [sp]; include_temperature=true,
                           include_tsfc=true, include_emissivity=true)
    # continuum ON (only the species' own continuum + relevant CIA, to keep it data-light)
    conts = sp == H2O ? (:h2o,) : (:co2,)
    fm = (iasi=iasi, apply_continuum=true, continua=conts, with_ils=true)
    fd = finite_difference_jacobian(prof, ll, spec; T_sfc=290.0, ε_sfc=0.98,
            observable=:bt,
            steps=default_fd_steps(spec; δT=0.1, δtsfc=0.1, δlogvmr=1e-3, δε=1e-3),
            fm_kwargs=fm)
    rel(an, cols) = maximum(abs.(an.K[:,cols] .- fd.K[:,cols])) /
                    (maximum(abs.(fd.K[:,cols])) + 1e-30)
    function jac(; vmr_coupling)
        analytic_jacobian(prof, ll, spec; iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
            observable=:bt, apply_continuum=true, continua=conts, with_ils=true,
            vmr_coupling=vmr_coupling)
    end
    vcol = spec.vmr_ranges[1][2]; tr = spec.temp_range
    for (lbl, an) in (("2b (no cont deriv)", jac(vmr_coupling=true)),)  # placeholder
    end
    # Compare 2c (current, full) — continuum derivatives are always on when apply_continuum.
    an = jac(vmr_coupling=true)
    @printf("%-8s  VMR rel=%.3e  T rel=%.3e  Tsfc=%.2e  ε=%.2e  y0=%.1e\n",
            tag, rel(an, vcol), rel(an, tr),
            maximum(abs.(an.K[:,spec.tsfc_index].-fd.K[:,spec.tsfc_index])),
            maximum(abs.(an.K[:,spec.emis_index].-fd.K[:,spec.emis_index])),
            maximum(abs.(an.y0 .- fd.y0)))
end

# H2O: continuum is quadratic in vmr; exercises VMR + T continuum derivatives.
h2o = [0.01*exp(-3.0*(1000.0-pk)/800.0) for pk in p]
run(H2O, Dict(H2O => HITRANLinelist([mkW(700.6,3.0e-21,200.0), mkW(702.3,1.2e-21,450.0)])),
    Dict(H2O=>h2o), "H2O")

# CO2: MT-CKD CO2 continuum (linear in vmr_co2).
run(CO2, Dict(CO2 => HITRANLinelist([mkC(700.5,2.0e-21,250.0), mkC(702.0,8.0e-22,600.0)])),
    Dict(CO2=>fill(4.0e-4,nlev)), "CO2")

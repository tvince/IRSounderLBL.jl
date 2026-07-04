# Phase 6.4 — does the LM-aware ∂σ/∂{T,p} make the T and VMR columns match FD
# with line mixing (VP_Y and VP_W) active?
using IRSounderLBL
using Printf

const RL = IRSounderLBL.RelmatLine
const RB = IRSounderLBL.RelmatBand
const WT = IRSounderLBL.W0B0Table
const RD = IRSounderLBL.HITRANRelmatData

# A small coupled CO2 Q-branch near 700 cm⁻¹ (synthetic, data-free). DipoT/PopuT0
# are scaled so the S-file intensity S0 = DipoT²·PopuT0·ν·stim ≈ 2e-24 matches the
# Voigt baseline lines below — i.e. the dispersive correction is a true PERTURBATION
# (~Y·p of the baseline), not a 24-orders-too-large term that saturates the column.
const _DIPO = 1.7e-13   # DipoT; with PopuT0=0.1 ⇒ S0 ≈ 2e-24 at ν≈700
function build_relmat()
    l1 = RL(Int8(1), 700.40, Int16(20), Int8(0), 0.09, Float32(0.75), Float32(-0.004),
            250.0, _DIPO, 0.10, _DIPO)
    l2 = RL(Int8(1), 700.55, Int16(18), Int8(0), 0.09, Float32(0.75), Float32(-0.004),
            240.0, _DIPO, 0.09, _DIPO)
    l3 = RL(Int8(1), 701.30, Int16(16), Int8(0), 0.085, Float32(0.75), Float32(-0.004),
            300.0, 0.9*_DIPO, 0.08, 0.9*_DIPO)
    band = RB("CO2Q", Int8(1), Int8(1), Int8(1), 700.4, 701.3, [l1, l2, l3])
    wt = WT()
    wt.qq[(20, 18)] = (log(0.035), 0.5)
    wt.qq[(20, 16)] = (log(0.025), 0.5)
    wt.qq[(18, 16)] = (log(0.030), 0.5)
    return RD([band], Dict{NTuple{2,Int8}, typeof(wt)}((Int8(1), Int8(1)) => wt))
end

iasi = IASIInstrument(699.0, 703.0, 0.5, 9, 2.0, 0.5)
nlev = 6
p = collect(range(1000.0, 200.0; length=nlev))
T = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
z = (1000.0 .- p) ./ 66.0

mkC(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                           Float32(0.09), Float32(0.08), E, Float32(0.75), Float32(-0.004))
# Full HITRAN Voigt baseline (the relmat S-files are NOT a complete linelist).
# Strengths ~1e-24 → a moderately thick (τ≈O(1)), unsaturated test band.
ll = Dict(CO2 => HITRANLinelist([mkC(700.40, 2.0e-24, 250.0), mkC(700.55, 1.8e-24, 240.0),
                                 mkC(701.30, 1.4e-24, 300.0), mkC(700.95, 9.0e-25, 500.0)]))
prof = AtmosphericProfile(p, T, z, Dict(CO2 => fill(4.0e-4, nlev)))

function run(tag, lm)
    spec = StateVectorSpec(nlev, [CO2]; include_temperature=true,
                           include_tsfc=true, include_emissivity=true)
    fm = (iasi=iasi, apply_continuum=false, with_ils=true, line_mixing=lm)
    fd = finite_difference_jacobian(prof, ll, spec; T_sfc=290.0, ε_sfc=0.98,
            observable=:bt,
            steps=default_fd_steps(spec; δT=0.1, δtsfc=0.1, δlogvmr=1e-3, δε=1e-3),
            fm_kwargs=fm)
    an = analytic_jacobian(prof, ll, spec; iasi=iasi, T_sfc=290.0, ε_sfc=0.98,
            observable=:bt, apply_continuum=false, with_ils=true, line_mixing=lm)
    rel(cols) = maximum(abs.(an.K[:, cols] .- fd.K[:, cols])) /
                (maximum(abs.(fd.K[:, cols])) + 1e-30)
    vcol = spec.vmr_ranges[1][2]; tr = spec.temp_range
    @printf("%-6s  VMR rel=%.3e  T rel=%.3e  Tsfc=%.2e  ε=%.2e  y0=%.2e\n",
            tag, rel(vcol), rel(tr),
            maximum(abs.(an.K[:, spec.tsfc_index] .- fd.K[:, spec.tsfc_index])),
            maximum(abs.(an.K[:, spec.emis_index] .- fd.K[:, spec.emis_index])),
            maximum(abs.(an.y0 .- fd.y0)))
end

relmat = build_relmat()
println("== Phase 6.4: LM-aware analytic Jacobian vs FD ==")
run("none",  nothing)
run("VP_Y",  VPYLineMixing(relmat))
run("VP_W",  VPWLineMixing(relmat; whitelist=Set(["CO2Q"])))

# Phase 5 — synthetic closed-loop optimal-estimation retrieval.
using IRSounderLBL
using LinearAlgebra
using Printf

mk(ν0, S, E) = HITRANLine(Int8(2), Int8(1), ν0, S, 1.0,
                          Float32(0.07), Float32(0.08), E,
                          Float32(0.75), Float32(-0.003))
ll = HITRANLinelist([mk(700.5, 2.0e-21, 250.0), mk(702.0, 8.0e-22, 600.0)])
linelists = Dict(CO2 => ll)
iasi = IASIInstrument(699.0, 705.0, 0.5, 13, 2.0, 0.5)
fm   = (iasi=iasi, apply_continuum=false, with_ils=true)

nlev = 6
p = collect(range(1000.0, 200.0; length=nlev))
Ttrue = 288.0 .+ (225.0 - 288.0) .* (1000.0 .- p) ./ 800.0
z = (1000.0 .- p) ./ 66.0
base = AtmosphericProfile(p, Ttrue, z, Dict(CO2 => fill(4.0e-4, nlev)))

# ── Test A: recover T_sfc alone (ε fixed & known; well-posed, noiseless) ──
let
    spec = StateVectorSpec(nlev, GasSpecies[]; include_temperature=false,
                           include_tsfc=true, include_emissivity=false)
    νg, R, ytrue = iasi_forward_model(base, linelists; T_sfc=298.0, ε_sfc=0.96, dptmn=0.0, fm...)
    xa = [288.0]                             # wrong first guess (10 K off)
    Sa = reshape([100.0], 1, 1)              # loose prior
    Se = Matrix(Diagonal(fill(0.2^2, length(ytrue))))
    # ε is fixed in the forward via... it is NOT in the state, so unpack_state sets ε→1.0;
    # generate ytrue with ε=1.0 to match the retrieval's fixed ε.
    νg, R, ytrue = iasi_forward_model(base, linelists; T_sfc=298.0, ε_sfc=1.0, dptmn=0.0, fm...)
    r = optimal_estimation(ytrue, spec, base, linelists; xa=xa, Sa=Sa, Se=Se,
                           fm_kwargs=fm, verbose=true)
    @printf("[A] conv=%s iters=%d  T_sfc: %.5f (true 298)  DOF=%.3f χ²=%.2e\n",
            r.converged, r.n_iter, r.x[1], r.dof, r.chi2)
end

# ── Test B: retrieve T_sfc + T(z) profile (ε fixed; noisy) — fit + diagnostics ──
let
    spec = StateVectorSpec(nlev, GasSpecies[]; include_temperature=true,
                           include_tsfc=true, include_emissivity=false)
    Ts_true = 296.0
    x_true = pack_state(spec, base; T_sfc=Ts_true, ε_sfc=1.0)
    νg, R, ytrue = iasi_forward_model(base, linelists; T_sfc=Ts_true, ε_sfc=1.0, dptmn=0.0, fm...)
    σ_noise = 0.2
    yobs = ytrue .+ σ_noise .* [sin(13.0*i) for i in 1:length(ytrue)]   # deterministic "noise"

    xa = copy(x_true); xa[spec.temp_range] .+= 4.0; xa[spec.tsfc_index] = 290.0
    Sa = Matrix(Diagonal([fill(5.0^2, nlev); 25.0]))             # T:5K, Tsfc:5K
    Se = Matrix(Diagonal(fill(σ_noise^2, length(ytrue))))
    r = optimal_estimation(yobs, spec, base, linelists; xa=xa, Sa=Sa, Se=Se,
                           fm_kwargs=fm, verbose=true)
    fit = maximum(abs.(r.y_fit .- yobs))
    @printf("[B] conv=%s iters=%d  T_sfc=%.3f (true 296)\n", r.converged, r.n_iter, r.x[spec.tsfc_index])
    @printf("    DOF=%.3f / n=%d  χ²=%.2f (n_y=%d)  cost %.3e→%.3e  max|y_fit-y|=%.3f K\n",
            r.dof, spec.n, r.chi2, length(ytrue), r.cost[1], r.cost[end], fit)
    @printf("    Ŝ symmetric? %s  posdef? %s\n",
            isapprox(r.S_hat, r.S_hat'), isposdef(Symmetric(r.S_hat)))
end

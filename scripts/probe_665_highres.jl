# Probe Julia VP_Y BT near 665 cm⁻¹ at native 0.005 cm⁻¹ to test whether the
# 33 K gap vs ARTS at the Q-branch peak is a downsampling artifact.
#
# If the 0.005 cm⁻¹ Julia BT also reads ~219 K at ν=665.00, the gap is *not*
# a resampling artifact — both systems sample the same value at exact grid
# points (`linear_interpolation` at a grid node returns the node value).
#
# Run:  julia --project -t auto scripts/probe_665_highres.jl

using IRSounderLBL
using Printf

const NU_MIN  = 660.0
const NU_MAX  = 670.0
const CUTOFF  = 25.0
const LM_DIR  = "data/Line-mixing_HITRAN2020/data_new"

t0 = time()
relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)

function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    return HITRANLinelist(all_lines)
end

ll_co2 = load_multi(1:3, NU_MIN, NU_MAX, "co2_645_2760")
ll_h2o = load_multi(1:3, NU_MIN, NU_MAX, "h2o_645_2760")
linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2, H2O => ll_h2o)
prof = afgl_us_standard_50lev()

# Run at native 0.005 cm⁻¹ (DNU_OUT = 0.005, HRF = 1).  IASI ν_min, ν_max set
# to the same window; n_ch chosen for 0.005 spacing.
function run_case(label, lm; dnu_out=0.005, hrf=1)
    n_ch = round(Int, (NU_MAX - NU_MIN) / dnu_out) + 1
    iasi = IASIInstrument(NU_MIN, NU_MAX, dnu_out, n_ch, 2.0, 0.5)
    ν, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = hrf,
                                   cutoff          = CUTOFF,
                                   apply_continuum = true,
                                   with_ils        = false,
                                   line_mixing     = lm)
    @printf("%-12s   %d channels @ %.3f cm⁻¹\n", label, length(BT), dnu_out)
    return ν.ν, BT
end

println("Running at native 0.005 cm⁻¹ (no downsampling) …")
ν1, BT_vpy_hi    = run_case("VP_Y hi-res", VPYLineMixing(relmat); dnu_out=0.005, hrf=1)
println("Running at native 0.25 cm⁻¹ (default downsampling) …")
ν2, BT_vpy_coarse= run_case("VP_Y coarse", VPYLineMixing(relmat); dnu_out=0.25,  hrf=50)
@printf("Total: %.1f s\n\n", time() - t0)

# Show values in the spike region
println("ν (cm⁻¹) :    BT_hi (0.005)    BT_coarse (0.25)")
for ν_query in [664.5, 664.75, 664.875, 664.95, 664.975, 665.0, 665.025, 665.05, 665.125, 665.25, 665.5]
    i_hi = argmin(abs.(ν1 .- ν_query))
    j_co = argmin(abs.(ν2 .- ν_query))
    @printf("%7.3f       %8.2f         %8.2f\n",
            ν_query, BT_vpy_hi[i_hi], BT_vpy_coarse[j_co])
end

# What does the high-res BT look like RIGHT around the 665.00 cm⁻¹ point?
println("\nFull high-res sweep 664.95-665.05 (every 0.005 cm⁻¹):")
mask = (ν1 .>= 664.95) .& (ν1 .<= 665.05)
for i in findall(mask)
    @printf("%7.4f  %.3f K\n", ν1[i], BT_vpy_hi[i])
end

# Compare to ARTS @ 665.00 = 253.0 K
println("\nARTS @ 665.00 cm⁻¹ = 253.0 K")
println("Julia VP_Y hi-res @ 665.00 cm⁻¹ = ", round(BT_vpy_hi[argmin(abs.(ν1 .- 665.0))], digits=2), " K")
println("Julia VP_Y coarse  @ 665.00 cm⁻¹ = ", round(BT_vpy_coarse[argmin(abs.(ν2 .- 665.0))], digits=2), " K")

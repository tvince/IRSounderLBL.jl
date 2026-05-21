"""
Smoke test: does the line-mixing wrapper run end-to-end for the full IASI
range (645–2760 cm⁻¹)?

Two checks:
  1. `load_hitran_relmat(LM_DIR, 645.0, 2760.0; stot_min=0.0)` succeeds and the
     band count / coverage is sane.
  2. `iasi_forward_model(...; line_mixing=VPYLineMixing(relmat))` and
     `VPWLineMixing(relmat)` run on a *small* IASI sub-grid (a 100 cm⁻¹ slice in
     the 4.3 µm CO₂ ν₃ region, the most-trafficked area outside 15 µm).

Not a validation run — just a "does it work?" check.

Run:
  julia --project -t auto scripts/smoke_lm_full_iasi.jl
"""

using RadiativeTransfer
using Printf

const NU_MIN  = 645.0
const NU_MAX  = 2760.0
const LM_DIR  = "data/Line-mixing_HITRAN2020/data_new"
const CUTOFF  = 25.0

t0 = time()
println("[1] Loading HITRAN relmat for the full IASI range $(NU_MIN)–$(NU_MAX) cm⁻¹ …")
@time relmat = load_hitran_relmat(LM_DIR, NU_MIN, NU_MAX; stot_min=0.0)
@printf("  bands loaded: %d\n", length(relmat.bands))

# Coverage by 100 cm⁻¹ buckets
buckets = Dict{Int,Int}()
for b in relmat.bands
    mid = round(Int, (b.ν_min + b.ν_max) / 2 / 100) * 100
    buckets[mid] = get(buckets, mid, 0) + 1
end
println("  band centres by 100 cm⁻¹ bucket (top 8):")
for (k, v) in sort(collect(buckets), by=x->-x[2])[1:min(8, length(buckets))]
    @printf("    %4d  %d bands\n", k, v)
end

# Strongest single bands across the whole range (for context)
band_S = Tuple{String, Float64, Int}[]
for b in relmat.bands
    S = 0.0
    for line in b.lines
        S += line.DipoT^2 * line.PopuT0 * line.ν
    end
    push!(band_S, (b.name, S, Int(b.isot)))
end
sort!(band_S, by=x->-x[2])
println("  strongest 5 bands (raw S, all isotopologues):")
for t in band_S[1:5]
    @printf("    %-13s  S≈%.3e   iso=%d\n", t[1], t[2], t[3])
end

println("\n[2] Smoke-testing forward model on a 100 cm⁻¹ slice in the 4.3 µm region …")
const SMOKE_MIN = 2300.0
const SMOKE_MAX = 2400.0
const DNU_OUT   = 0.25
const HRF       = 50

n_ch = round(Int, (SMOKE_MAX - SMOKE_MIN) / DNU_OUT) + 1
iasi = IASIInstrument(SMOKE_MIN, SMOKE_MAX, DNU_OUT, n_ch, 2.0, 0.5)

function load_multi(iso_ids, ν_min, ν_max, base)
    all_lines = HITRANLine[]
    for iso_id in iso_ids
        fname = iso_id == 1 ? "$(base).par" : "$(base)_iso$(iso_id).par"
        fpath = joinpath("data", fname)
        isfile(fpath) || continue
        ll = load_hitran_par(fpath; ν_min=ν_min - CUTOFF, ν_max=ν_max + CUTOFF)
        append!(all_lines, ll.lines)
    end
    isempty(all_lines) && return HITRANLinelist(all_lines, Set{Int}(), ν_min, ν_max)
    return HITRANLinelist(all_lines)
end

# The existing data files cover 645–2760 cm⁻¹ already
ll_co2 = load_multi(1:3, SMOKE_MIN, SMOKE_MAX, "co2_645_2760")
ll_h2o = load_multi(1:3, SMOKE_MIN, SMOKE_MAX, "h2o_645_2760")
@printf("  CO2: %d lines   H2O: %d lines  (in %.0f–%.0f cm⁻¹)\n",
        length(ll_co2), length(ll_h2o), SMOKE_MIN, SMOKE_MAX)

linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll_co2, H2O => ll_h2o)
prof = afgl_us_standard_50lev()

for (label, lm) in (
    ("no LM",       nothing),
    ("VPYLineMixing", VPYLineMixing(relmat)),
    ("VPWLineMixing (default)", VPWLineMixing(relmat)),
)
    t1 = time()
    ν, _, BT = iasi_forward_model(prof, linelists;
                                   iasi            = iasi,
                                   high_res_factor = HRF,
                                   cutoff          = CUTOFF,
                                   apply_continuum = true,
                                   apply_ils       = false,
                                   line_mixing     = lm)
    elapsed = time() - t1
    finite = all(isfinite, BT)
    @printf("  %-26s %5.1f s   BT range %.1f–%.1f K   all-finite=%s\n",
            label, elapsed, minimum(BT), maximum(BT), finite)
end

@printf("\nTotal smoke-test time: %.1f s (%d threads)\n", time() - t0, Threads.nthreads())

"""
First-order Rosenkranz line mixing for CO2 bands.

Implements the VP_Y approximation from Lamouroux et al. (2015) JQSRT 151:88-96
and Hashemi et al. (2020) JQSRT 256:107283.  Uses the HITRAN 2020 relaxation
matrix data (BandInfo.dat, S*.dat, WTfit*.dat).

Line mixing redistributes absorption between coupled lines.  We split this
into two terms that don't interfere:

    σ_CO2(ν) = σ_voigt(ν) + σ_disp(ν)

  σ_voigt(ν) = Σ over ALL HITRAN CO2 lines of  Sᵢ·(f/√π)·Re[w(zᵢ)]
               (computed by `compute_voigt_cross_sections` with `ll_co2`)
  σ_disp(ν)  = Σ over S-file LM lines of       Yᵢ·p·Sᵢ·(f/√π)·Im[w(zᵢ)]
               with Fortran clip |Yᵢ·p| ≤ 0.08 (else Yᵢ → 0 for that line)

The dispersive perturbation can be locally negative — it's a redistribution
of intensity, not a new absorber.  σ_voigt is the positive baseline.

S-files cover only bands in BandInfo.dat that pass the load filters
(li≤8, |li−lf|≤1); they are NOT a complete CO2 line database.  Using them
for the Re[w] sum would undercount absorption, which is why the baseline
must come from the full HITRAN linelist.
"""

using SpecialFunctions: erfcx
using LinearAlgebra: eigen, inv

const _T0_LM   = 296.0   # HITRAN reference temperature (K)
const _CT_LM   = 1.4388  # hc/kB (cm·K) for stimulated emission factor

# ── Data structures ───────────────────────────────────────────────────────────

struct RelmatLine
    isot::Int8
    ν::Float64        # line centre (cm⁻¹)
    Ji::Int16         # initial J quantum number
    branch::Int8      # -1=P, 0=Q, +1=R
    gV_air::Float64   # air-broadened HWHM at 296 K (cm⁻¹/atm)
    n_air::Float32    # temperature exponent for air broadening
    shift::Float32    # pressure shift (cm⁻¹/atm)
    E_lower::Float64  # lower-state energy (cm⁻¹)
    Dipo0::Float64    # rigid-rotor dipole (cm/molec^0.5) at 296 K
    PopuT0::Float64   # lower-state population at 296 K
    DipoT::Float64    # temperature-corrected dipole at 296 K
end

"""
    BandCoupling

T-independent sparse coupling structure of a `RelmatBand` for a given WTfit
table: every ordered line pair (i, j) — original `band.lines` order, outer i,
inner j with `Ji_j ≤ Ji_i` — that passes the Fortran pair skip, the
isotopologue-symmetry restriction, and has a WTfit entry, together with its
(W0R, B0R) coefficients. CSR-grouped by the outer line: the pairs of line `i`
are `row_ptr[i]:(row_ptr[i+1]-1)`.

Precomputing this removes the `Symbol`/`getfield`/`Dict` lookups from the
O(n²) loop in `_build_W_matrix`; each temperature evaluation is then just
`exp(W0R − B0R·log(T0/T))` over the pair list. Built once per band (eagerly by
`load_hitran_relmat`, lazily by `_band_coupling` for directly-constructed
bands) — valid for all T, but tied to the band's own WTfit table.
"""
struct BandCoupling
    row_ptr::Vector{Int32}
    col::Vector{Int32}
    W0R::Vector{Float64}
    B0R::Vector{Float64}
end

struct RelmatBand
    name::String
    li::Int8          # vibrational angular momentum, lower state
    lf::Int8          # vibrational angular momentum, upper state
    isot::Int8
    ν_min::Float64
    ν_max::Float64
    lines::Vector{RelmatLine}
    molecule::Int8    # HITRAN molecule id (2=CO2, 6=CH4) — selects mass/abundance/Q(T)
    coupling::Base.RefValue{Union{Nothing, BandCoupling}}  # lazy T-independent pair table
end
# Back-compat: existing 7/8-arg call sites; the coupling table starts empty.
RelmatBand(name, li, lf, isot, ν_min, ν_max, lines, molecule) =
    RelmatBand(name, li, lf, isot, ν_min, ν_max, lines, molecule,
               Ref{Union{Nothing, BandCoupling}}(nothing))
RelmatBand(name, li, lf, isot, ν_min, ν_max, lines) =
    RelmatBand(name, li, lf, isot, ν_min, ν_max, lines, Int8(2))

# W0/B0 parameters keyed by (Ji, Jip) for each of 9 branch-pair combinations.
# W(T) = exp(W0) × (T/T0)^B0  [off-diagonal element in cm⁻¹/atm]
struct W0B0Table
    pp::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    pq::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    pr::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    qp::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    qq::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    qr::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    rp::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    rq::Dict{NTuple{2,Int}, NTuple{2,Float64}}
    rr::Dict{NTuple{2,Int}, NTuple{2,Float64}}
end

W0B0Table() = W0B0Table(Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict())

struct HITRANRelmatData
    bands::Vector{RelmatBand}
    # key: (lli, llf) = (min(li,lf), max(li,lf))
    wtfit::Dict{NTuple{2,Int8}, W0B0Table}
    species::GasSpecies   # which gas this relmat applies to (dispatch gate)
end
# Back-compat: existing 2-arg call sites (all CO2) default species=CO2.
HITRANRelmatData(bands, wtfit) = HITRANRelmatData(bands, wtfit, CO2)

# ── Line-mixing dispatch wrappers ─────────────────────────────────────────────
#
# Pass an `AbstractLineMixing` subtype to `iasi_forward_model(..., line_mixing=…)`
# to enable line mixing for CO2.  `nothing` (the default) leaves the existing
# pure-Voigt path untouched.  Dispatch is on the wrapper *type* — VP_W will get
# its own struct because it needs to carry cached eigendecompositions per (T,p).
"""
    AbstractLineMixing

Marker type for CO2 line-mixing models passed to `iasi_forward_model`.
Concrete subtypes: [`VPYLineMixing`](@ref).
"""
abstract type AbstractLineMixing end

"""
    VPYLineMixing(data::HITRANRelmatData; min_band_strength=0.0)

First-order Rosenkranz (Voigt + dispersive perturbation) line mixing.
Pass to `iasi_forward_model(...; line_mixing=VPYLineMixing(relmat))` to enable
LM on the CO2 channel.  See `compute_voigt_lm_cross_sections`.

`min_band_strength` (#4 cutoff): skip bands whose integrated intensity
(`_band_eff_strength`, T_REF) is below this value — drops negligibly-weak
high-isotopologue/hot bands. `0.0` (default) processes all loaded bands (exact).
Recommended opt-in value `1e-25`: ~1.4× end-to-end speedup at ≤0.6 mK max|ΔBT|
(15 µm 0.33 mK / 4 µm 0.56 mK, both well under 1 mK; drops ~50% of bands —
the weak ones with few lines). `1e-26` is ~1.1-1.2× at ≤0.09 mK if you want a
near-lossless trim. See scripts/sweep_band_cutoff.jl for the BT/timing sweep.
"""
struct VPYLineMixing <: AbstractLineMixing
    data::HITRANRelmatData
    min_band_strength::Float64
end
VPYLineMixing(data::HITRANRelmatData; min_band_strength::Float64 = 0.0) =
    VPYLineMixing(data, min_band_strength)

"""
    VPWLineMixing(data; n_top=5, isotopes=nothing, ν_window=nothing, whitelist=nothing)

Full-matrix line mixing for CO₂.  Bands in `whitelist` receive the
eigendecomposition treatment (`band_modes` + `compute_vpw_band_xsec`); all
other bands fall back to the cheaper VP_Y dispersive perturbation.  The full
HITRAN CO₂ linelist is always evaluated as the Voigt baseline; LM contributes
only as an additive perturbation.

If `whitelist` is `nothing` (default), the whitelist is built by
[`default_vpw_whitelist`]: when `isotopes === nothing`, the top-`n_top` bands
overall are selected, weighted by `S(T_REF) × natural isotopologue abundance`;
when `isotopes` is given (e.g. `[1, 2]`), the top-`n_top` bands per isotope
(by raw `S(T_REF)`) are picked.  When `ν_window=(ν_lo, ν_hi)` is given, only
bands whose [ν_min, ν_max] overlaps the window are considered for scoring —
useful when `data` was loaded over a larger range than the forward model
will evaluate (e.g. full IASI relmat but a 15 µm-only run).

# Fields
- `data`:      `HITRANRelmatData` (relaxation-matrix bands + WTfit tables)
- `whitelist`: band names processed via VP_W; the rest use VP_Y
- `min_band_strength`: #4 cutoff — skip bands below this integrated intensity
  (`_band_eff_strength`, T_REF). `0.0` (default) = exact. Recommended opt-in `1e-25`:
  ~1.4× end-to-end at ≤0.6 mK max|ΔBT| (drops ~50% of weak bands); `1e-26` ~1.1-1.2×
  at ≤0.09 mK. See scripts/sweep_band_cutoff.jl.
"""
struct VPWLineMixing <: AbstractLineMixing
    data::HITRANRelmatData
    whitelist::Set{String}
    min_band_strength::Float64
end

function VPWLineMixing(data::HITRANRelmatData;
                       n_top::Int = 5,
                       isotopes::Union{Nothing, Vector{Int}} = nothing,
                       ν_window::Union{Nothing, Tuple{Real,Real}} = nothing,
                       whitelist::Union{Nothing, Set{String}} = nothing,
                       min_band_strength::Float64 = 0.0)
    wl = isnothing(whitelist) ?
         default_vpw_whitelist(data; n_top=n_top, isotopes=isotopes, ν_window=ν_window) :
         whitelist
    return VPWLineMixing(data, wl, min_band_strength)
end

# Per-species cross-section dispatch used by `iasi_forward_model`.
# `nothing` and any non-CO2 species fall through to the plain Voigt path.
# `VPYLineMixing` + CO2 routes through the Voigt+dispersive wrapper.
function _species_cross_section(::Nothing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend,
                                 lm_cutoff::Float64 = cutoff)
    compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                  vmr_self=vmr_self, cutoff=cutoff, backend=backend)
end

function _species_cross_section(lm::AbstractLineMixing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend,
                                 lm_cutoff::Float64 = cutoff)
    compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                  vmr_self=vmr_self, cutoff=cutoff, backend=backend)
end

function _species_cross_section(lm::VPYLineMixing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend,
                                 lm_cutoff::Float64 = cutoff)
    sp == lm.data.species || return compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                                      vmr_self=vmr_self, cutoff=cutoff, backend=backend)
    compute_voigt_lm_cross_sections(ν_grid, ll, lm.data, T, p_atm; cutoff=cutoff,
                                     lm_cutoff=lm_cutoff, min_band_strength=lm.min_band_strength)
end

function _species_cross_section(lm::VPWLineMixing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend,
                                 lm_cutoff::Float64 = cutoff)
    sp == lm.data.species || return compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                                      vmr_self=vmr_self, cutoff=cutoff, backend=backend)
    compute_voigt_vpw_cross_sections(ν_grid, ll, lm.data, T, p_atm;
                                       whitelist=lm.whitelist, cutoff=cutoff,
                                       lm_cutoff=lm_cutoff, min_band_strength=lm.min_band_strength)
end

# ── LM-only perturbation accessor (for the Jacobian ∂σ/∂{T,p}, roadmap §6.4) ──
# Returns Δσ = σ_LM − σ_voigt for the active model: the additive line-mixing term
# the Jacobian central-differences in (T, p). VP_Y → dispersive sum; VP_W → the
# eigendecomposed band-perturbation sum. The Voigt baseline keeps its analytic grad;
# only this (cheap relative to the baseline) perturbation is finite-differenced.

"""
    LMTCache

Fixed-temperature cache of the O(n²) per-band quantities the LM perturbation
rebuilds identically across same-T calls (lever 3): `Ys` = per-band Rosenkranz
Y(T) for the dispersive path, `Wρ` = per-band-index (W, PopuT) for VP_W
whitelisted bands. Neither carries pressure, so a cache built at the Jacobian's
unperturbed T serves the centre call and both p-perturbation calls.
"""
struct LMTCache
    T::Float64
    Ys::Vector{Union{Nothing, Vector{Float64}}}
    Wρ::Dict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}
end

_lm_T_cache(lm::VPYLineMixing, T::Float64) =
    LMTCache(T, _lm_dispersive_Ys(lm.data, T; min_band_strength=lm.min_band_strength),
             Dict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}())

function _lm_T_cache(lm::VPWLineMixing, T::Float64)
    Ys = _lm_dispersive_Ys(lm.data, T; min_band_strength=lm.min_band_strength,
                           skip=lm.whitelist)
    Wρ = Dict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}()
    for (bi, band) in enumerate(lm.data.bands)
        band.name in lm.whitelist || continue
        Int(band.li) > 8 && continue
        lm.min_band_strength > 0.0 && _band_eff_strength(band) < lm.min_band_strength && continue
        wtfit = get(lm.data.wtfit, (Int8(min(band.li, band.lf)), Int8(max(band.li, band.lf))), nothing)
        wtfit === nothing && continue
        Wρ[bi] = _build_W_matrix(band, wtfit, T)
    end
    return LMTCache(T, Ys, Wρ)
end

function _lm_perturbation(lm::VPYLineMixing, ν_grid, T, p_atm; cutoff,
                          cache::Union{Nothing, LMTCache} = nothing)
    @assert cache === nothing || cache.T == T "LMTCache built at a different T"
    compute_lm_dispersive_correction(ν_grid, lm.data, T, p_atm;
                                     cutoff=cutoff, min_band_strength=lm.min_band_strength,
                                     Ys = cache === nothing ? nothing : cache.Ys)
end
function _lm_perturbation(lm::VPWLineMixing, ν_grid, T, p_atm; cutoff,
                          cache::Union{Nothing, LMTCache} = nothing)
    @assert cache === nothing || cache.T == T "LMTCache built at a different T"
    _vpw_perturbation(ν_grid, lm.data, T, p_atm; whitelist=lm.whitelist,
                      cutoff=cutoff, min_band_strength=lm.min_band_strength,
                      Ys = cache === nothing ? nothing : cache.Ys,
                      Wρs = cache === nothing ? nothing : cache.Wρ)
end

# ── File parsers ──────────────────────────────────────────────────────────────

function _parse_fortran_d(s::AbstractString)::Float64
    parse(Float64, replace(strip(s), r"[Dd]" => "e"))
end

"""
    _parse_bandinfo(basedir, ν_min, ν_max; stot_min) -> Vector{NamedTuple}

Parse BandInfo.dat and return entries whose wavenumber range overlaps [ν_min, ν_max]
and whose total band intensity ≥ stot_min.

BandInfo format (Fortran):
  Format(I1,2(A2,i1,A2),a2,E12.4,2(1x,f12.6),8x,3I4)
  → isot c1 lf C2 c3 li c4 c5  Stot  SgMin  SgMax  jmxP jmxQ jmxR

Column layout (1-indexed):
  1     : isot
  2-3   : c1 (A2)
  4     : lf (i1)
  5-6   : C2 (A2)
  7-8   : c3 (A2)
  9     : li (i1)
  10-11 : c4 (A2)
  12-13 : c5 (A2)
  14-25 : Stot (E12.4)
  26    : skip (1x)
  27-38 : SgMin (f12.6)
  39    : skip (1x)
  40-51 : SgMax (f12.6)
  52-59 : skip (8x)
  60-63 : jmxP (I4)
  64-67 : jmxQ (I4)
  68-71 : jmxR (I4)
"""
function _parse_bandinfo(basedir::String, ν_min::Float64, ν_max::Float64;
                          stot_min::Float64 = 1e-28)
    path = joinpath(basedir, "BandInfo.dat")
    result = NamedTuple[]
    open(path) do fh
        for raw in eachline(fh)
            line = rstrip(raw, ['\r', '\n', ' '])   # strip CRLF and trailing spaces
            length(line) < 71 && continue
            try
                isot = parse(Int8, line[1:1])
                isot == 0 && (isot = Int8(10))
                lf   = parse(Int8, line[4:4])
                li   = parse(Int8, line[9:9])
                Stot = parse(Float64, line[14:25])
                Stot < stot_min && continue
                sg_min = parse(Float64, line[27:38])
                sg_max = parse(Float64, line[40:51])
                sg_min > ν_max && continue
                sg_max < ν_min && continue
                jmxP = parse(Int, line[60:63])
                jmxQ = parse(Int, line[64:67])
                jmxR = parse(Int, line[68:71])
                # Band filename prefix: S + 13-char identifier (includes trailing spaces)
                band_id = rstrip(line[1:13])
                sfile   = "S" * band_id
                # Fortran skip trick: bypass LM for sparse bands
                li_eff = (jmxP < 40 || jmxR < 40 || (jmxQ >= 0 && jmxQ < 40)) ? Int8(55) : li
                push!(result, (; sfile, li=li_eff, lf, isot, ν_min=sg_min, ν_max=sg_max, Stot, jmxP, jmxQ, jmxR))
            catch
                continue
            end
        end
    end
    return result
end

"""
    _parse_sfile(path) -> Vector{RelmatLine}

Parse a S*.dat spectroscopic file.  Fortran format 1001:
  (2x,I1,F12.6,E10.3,10x,2(2f5.4,f4.2),f10.4,2f4.2,f8.6,50x,A1,I3,21x,(2f5.4,f4.2),f5.3,2d16.8)

Column layout (1-indexed):
  3       : isot
  4-15    : ν (cm⁻¹)
  16-25   : S (intensity)
  36-40   : gV_air
  64-73   : E_lower
  74-77   : n_air
  82-89   : shift
  140     : branch (P/Q/R)
  141-143 : Ji
  184-199 : Dipo0 (D-notation)
  200-215 : PopuT0 (D-notation)
"""
function _parse_sfile(path::String)::Vector{RelmatLine}
    lines = RelmatLine[]
    open(path) do fh
        for line in eachline(fh)
            length(line) < 215 && continue
            try
                isot_c = line[3]
                isot   = isot_c == '0' ? Int8(10) : Int8(isot_c - '0')
                ν      = parse(Float64, line[4:15])
                S      = parse(Float64, line[16:25])
                gV_air = parse(Float64, line[36:40])
                E_lower= parse(Float64, line[64:73])
                n_air  = parse(Float32, line[74:77])
                shift  = parse(Float32, line[82:89])
                br     = line[140]
                Ji     = parse(Int16, strip(line[141:143]))
                Dipo0  = _parse_fortran_d(line[184:199])
                PopuT0 = _parse_fortran_d(line[200:215])

                branch = br == 'P' ? Int8(-1) : (br == 'R' ? Int8(1) : Int8(0))

                stim = 1.0 - exp(-_CT_LM * ν / _T0_LM)
                DipoT = sqrt(max(0.0, abs(S) / (abs(PopuT0) * ν * stim)))

                push!(lines, RelmatLine(isot, ν, Ji, branch, gV_air, n_air,
                                        shift, E_lower, Dipo0, PopuT0, DipoT))
            catch
                continue
            end
        end
    end
    return lines
end

"""
    _load_wtfit(basedir, lli, llf) -> W0B0Table

Parse WTfit{lli}{llf}.dat.  Fortran format: (2d20.12,2f14.6,4I4)
  W0R, B0R: log W₀ and temperature exponent; W(T) = exp(W0R) × (T/T0)^B0R
  jic,jfc,jipc,jfpc: J quantum numbers determining branch-pair sub-table.
"""
function _load_wtfit(basedir::String, lli::Int, llf::Int)::W0B0Table
    fname = joinpath(basedir, "WTfit$(lli)$(llf).dat")
    isfile(fname) || return W0B0Table()
    tbl = W0B0Table()
    open(fname) do fh
        for raw in eachline(fh)
            line = rstrip(raw, ['\r', '\n'])
            # Format (Fortran): 2d20.12, 2f14.6, 4I4
            # col 1-20: W0R, 21-40: B0R, 41-54: dmaxDT, 55-68: WTmax
            # col 69-72: jic, 73-76: jfc, 77-80: jipc, 81-84: jfpc
            length(line) < 84 && continue
            try
                W0R  = _parse_fortran_d(line[1:20])
                B0R  = _parse_fortran_d(line[21:40])
                jic  = parse(Int, strip(line[69:72]))
                jfc  = parse(Int, strip(line[73:76]))
                jipc = parse(Int, strip(line[77:80]))
                jfpc = parse(Int, strip(line[81:84]))

                key = (jic, jipc)
                val = (W0R, B0R)

                # Branch types: P: j>jf, Q: j==jf, R: j<jf
                type_i  = jic  > jfc  ? :p : (jic  == jfc  ? :q : :r)
                type_ip = jipc > jfpc ? :p : (jipc == jfpc ? :q : :r)

                sub = Symbol(type_i, type_ip)   # :pp, :pq, ... :rr
                getfield(tbl, sub)[key] = val
            catch
                continue
            end
        end
    end
    return tbl
end

# ── Public loader ─────────────────────────────────────────────────────────────

"""
    load_hitran_relmat(basedir, ν_min, ν_max; stot_min=1e-28) -> HITRANRelmatData

Load HITRAN 2020 relaxation matrix data for CO2 bands overlapping
[ν_min, ν_max] (cm⁻¹).  `basedir` is the path to the `data_new/` directory.
`stot_min` is the minimum total band intensity (cm⁻¹/(molec·cm⁻²)) — bands
below this threshold contribute negligibly and are skipped.
"""
function load_hitran_relmat(basedir::String, ν_min::Float64, ν_max::Float64;
                             stot_min::Float64 = 1e-28)::HITRANRelmatData
    band_infos = _parse_bandinfo(basedir, ν_min, ν_max; stot_min)

    bands  = RelmatBand[]
    loaded_wtfit = Dict{NTuple{2,Int8}, W0B0Table}()

    for bi in band_infos
        li = Int(bi.li)
        lf = Int(bi.lf)

        # CalcW bypasses bands where li > 8, lf > 8, or |li−lf| > 1.
        # Those bands produce Y=0 so skip loading their S-files entirely.
        if li > 8 || lf > 8 || abs(li - lf) > 1
            continue
        end

        sfile = joinpath(basedir, bi.sfile * "  .dat")   # trailing spaces in filename
        isfile(sfile) || (sfile = joinpath(basedir, bi.sfile * ".dat"))
        isfile(sfile) || continue

        lines = _parse_sfile(sfile)
        isempty(lines) && continue

        push!(bands, RelmatBand(bi.sfile, bi.li, bi.lf, bi.isot, bi.ν_min, bi.ν_max, lines))

        lli = Int8(min(bi.li, bi.lf))
        llf = Int8(max(bi.li, bi.lf))
        key = (lli, llf)
        if !haskey(loaded_wtfit, key)
            loaded_wtfit[key] = _load_wtfit(basedir, Int(lli), Int(llf))
        end
        # Eagerly build the T-independent coupling table (lever 1) so the hot
        # path never has to (and threaded callers never race on the lazy Ref).
        _band_coupling(bands[end], loaded_wtfit[key])
    end

    return HITRANRelmatData(bands, loaded_wtfit)
end

# ── Y coefficient computation (CalcW in Fortran) ─────────────────────────────

"""
    _build_coupling(band, wtfit) -> BandCoupling

Derive the T-independent pair structure of `band` (see [`BandCoupling`](@ref)):
replay the `_build_W_matrix` pair loop's skips — `Ji_j ≤ Ji_i` (Fortran),
isotopologue symmetry, WTfit-entry presence — once, in original line order,
and record each surviving ordered pair's (j, W0R, B0R) grouped by outer i.
"""
function _build_coupling(band::RelmatBand, wtfit::W0B0Table)::BandCoupling
    n  = length(band.lines)
    li = Int(band.li)
    lf = Int(band.lf)
    isot = Int(band.isot)
    sym_restrict = isot > 2 && isot != 7 && isot != 10

    Ji = Int[line.Ji for line in band.lines]
    br = Int[line.branch for line in band.lines]

    row_ptr = Vector{Int32}(undef, n + 1)
    col = Int32[]; W0R = Float64[]; B0R = Float64[]
    for i in 1:n
        row_ptr[i] = Int32(length(col) + 1)
        jji = Ji[i]; jjf = jji + br[i]
        for j in 1:n
            j == i && continue
            jjip = Ji[j]; jjfp = jjip + br[j]

            # Fortran skip: only process pairs where Ji_j ≤ Ji_i
            jjip > jji && continue

            # Asymmetric isotopologue symmetry restriction
            sym_restrict && abs(jji - jjip) % 2 != 0 && continue

            # WTfit convention: if li > lf swap J roles for table lookup
            if li <= lf
                ji_eff = jji; jf_eff = jjf; jip_eff = jjip; jfp_eff = jjfp
            else
                ji_eff = jjf; jf_eff = jji; jip_eff = jjfp; jfp_eff = jjip
            end
            key = (ji_eff, jip_eff)

            type_i  = ji_eff  > jf_eff  ? :p : (ji_eff  == jf_eff  ? :q : :r)
            type_ip = jip_eff > jfp_eff ? :p : (jip_eff == jfp_eff ? :q : :r)

            v = get(getfield(wtfit, Symbol(type_i, type_ip)), key, nothing)
            v === nothing && continue

            push!(col, Int32(j)); push!(W0R, v[1]); push!(B0R, v[2])
        end
    end
    row_ptr[n + 1] = Int32(length(col) + 1)
    return BandCoupling(row_ptr, col, W0R, B0R)
end

# Memoized accessor. The cache is tied to the band's own WTfit table — every
# call site pairs a band with `relmat.wtfit[(min(li,lf), max(li,lf))]`, so the
# association is stable. `load_hitran_relmat` fills it eagerly; the lazy branch
# covers directly-constructed bands (tests/scripts) and is idempotent.
function _band_coupling(band::RelmatBand, wtfit::W0B0Table)::BandCoupling
    c = band.coupling[]
    c === nothing || return c
    c = _build_coupling(band, wtfit)
    band.coupling[] = c
    return c
end

"""
    _build_W_matrix(band, wtfit, T) -> (W, PopuT)

Build the sum-rule-corrected relaxation matrix W (cm⁻¹/atm) and the
temperature-scaled lower-state populations PopuT for a CO₂ band at
temperature T.  Both arrays are returned in *original* `band.lines` order.

W has:
  - Diagonal entries = air-broadened HWHM at T, p=1 atm.
  - Off-diagonal entries = sum-rule-corrected negative coupling strengths.
  - Zero in slots forbidden by isotopologue symmetry or absent from WTfit.

The sort-by-intensity + sum-rule renormalisation follows `CalcW` in
`LM_calc_CO2.for`.  Permuted back to the original line order so callers can
index by `band.lines[i]`.

This matrix is shared between VP_Y (Yᵢ = 2·Σⱼ |Dⱼ|/|Dᵢ|·Wⱼᵢ/(νᵢ−νⱼ)) and
VP_W (eigendecomposition of D + i·p·W).
"""
function _build_W_matrix(band::RelmatBand, wtfit::W0B0Table, T::Float64)
    n = length(band.lines)
    n == 0 && return zeros(Float64, 0, 0), Float64[]

    li = Int(band.li)
    lf = Int(band.lf)

    iso = Int(band.isot == 10 ? 10 : band.isot)
    RatioPart = partition_function(Int(band.molecule), iso, T_REF) / partition_function(Int(band.molecule), iso, T)
    PopuT = Vector{Float64}(undef, n)
    for (i, line) in enumerate(band.lines)
        PopuT[i] = line.PopuT0 * RatioPart *
                   exp(-_CT_LM * line.E_lower * (1.0/T - 1.0/_T0_LM))
    end

    # Decoupled bands (li>8 or |li-lf|>1) get diagonal-only W (γ_L per line).
    if li > 8 || lf > 8 || abs(li - lf) > 1
        W = zeros(Float64, n, n)
        for (i, line) in enumerate(band.lines)
            W[i, i] = Float64(line.gV_air) * (_T0_LM / T)^Float64(line.n_air)
        end
        return W, PopuT
    end

    DipoT_orig = Float64[line.DipoT for line in band.lines]
    ν_orig     = Float64[line.ν     for line in band.lines]
    S_eff      = [ν_orig[i] * PopuT[i] * DipoT_orig[i]^2 for i in 1:n]

    # The intensity sort is T-dependent (via PopuT). It matters twice: the
    # sum-rule renormalisation splits by sorted position, and equal-J pairs are
    # written from both directions with last-writer-wins semantics — so the pair
    # loop's OUTER order must replay the sorted order (inner order is free: each
    # outer iteration writes disjoint slots).
    ord = sortperm(S_eff, rev=true)

    cp = _band_coupling(band, wtfit)   # T-independent sparse pair table (lever 1)

    # Build W directly in original line order (replaces sorted-space build +
    # permute-back; identical values, same overwrite semantics).
    W = zeros(Float64, n, n)
    dlgT0T = log(_T0_LM / T)
    isot   = Int(band.isot)

    @inbounds for ir in 1:n
        i = ord[ir]
        for t in cp.row_ptr[i]:(cp.row_ptr[i+1] - Int32(1))
            j = Int(cp.col[t])
            ycal = exp(cp.W0R[t] - cp.B0R[t] * dlgT0T)
            W[j, i] = ycal
            W[i, j] = ycal * PopuT[i] / PopuT[j]
        end
    end

    # Off-diagonals negative (energy flows out of each state)
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        W[i, j] = -abs(W[i, j])
    end

    # Diagonal = air-broadened HWHM at T, p=1 atm
    for (i, line) in enumerate(band.lines)
        W[i, i] = Float64(line.gV_air) * ((_T0_LM / T)^Float64(line.n_air))
    end

    # Sum-rule renormalisation (Fortran convention: lower triangle, sorted order;
    # accessed through `ord` since W is stored in original order)
    Ji = Int[line.Ji for line in band.lines]
    Dipo0 = Float64[line.Dipo0 for line in band.lines]
    sym_restrict = isot > 2 && isot != 7 && isot != 10
    @inbounds for ir in 1:n
        i = ord[ir]
        Ji_i = Ji[i]
        sumLW = 0.0
        sumUp = 0.0
        for irp in 1:n
            j = ord[irp]
            sym_restrict && abs(Ji_i - Ji[j]) % 2 != 0 && continue
            if irp > ir
                sumLW += abs(Dipo0[j]) * W[j, i]
            else
                sumUp += abs(Dipo0[j]) * W[j, i]
            end
        end

        sumLW == 0.0 && continue
        ratio = -sumUp / sumLW
        for irp in (ir+1):n
            j = ord[irp]
            W[j, i] = W[j, i] * ratio
            W[i, j] = W[j, i] * PopuT[i] / PopuT[j]
        end
    end

    return W, PopuT
end

"""
    _calc_W_and_Y(band, wtfit, T) -> Vector{Float64}

First-order Rosenkranz Y coefficient (per atm) for each line in `band` at
temperature T.  Wraps `_build_W_matrix` and computes
  Yᵢ = 2 × Σⱼ≠ᵢ |Dⱼ|/|Dᵢ| × Wⱼᵢ / (νᵢ − νⱼ)
in original line order.  Diagonal entries of W are γ_L; the j==i skip in the
sum makes the Y computation invariant to them.
"""
function _calc_W_and_Y(band::RelmatBand, wtfit::W0B0Table, T::Float64)::Vector{Float64}
    n = length(band.lines)
    n == 0 && return Float64[]

    li = Int(band.li); lf = Int(band.lf)
    if li > 8 || lf > 8 || abs(li - lf) > 1
        return zeros(Float64, n)
    end

    W, _ = _build_W_matrix(band, wtfit, T)

    DipoT = Float64[line.DipoT for line in band.lines]
    ν     = Float64[line.ν     for line in band.lines]

    Y = zeros(Float64, n)
    for i in 1:n
        s = 0.0
        for j in 1:n
            j == i && continue
            Δν = ν[i] - ν[j]
            abs(Δν) < 1e-4 && (Δν = (Δν >= 0.0 ? 1.0 : -1.0) * 1e-4)
            s += 2.0 * abs(DipoT[j]) / abs(DipoT[i]) * (1.0 / Δν) * W[j, i]
        end
        Y[i] = s
    end
    return Y
end

# ── Per-species isotopologue molar masses (amu) for the Doppler width ────────
# Keyed by HITRAN molecule id, then isotopologue id.

const _CO2_MASS_AMU = Dict(
    1 => 43.98983, 2 => 44.99318, 3 => 45.99398,
    4 => 44.99403, 5 => 45.99706, 6 => 44.99706,
    7 => 47.99832, 8 => 47.00134, 9 => 48.00196,
    10=> 47.00427,
)
# CH4 isotopologue masses (amu): 1=¹²CH₄, 2=¹³CH₄, 3=¹²CH₃D, 4=¹³CH₃D.
const _CH4_MASS_AMU = Dict(
    1 => 16.031300, 2 => 17.034655, 3 => 17.037475, 4 => 18.040830,
)
const _LM_MASS_AMU = Dict(2 => _CO2_MASS_AMU, 6 => _CH4_MASS_AMU)

const _CTGAMD = 3.5812e-7   # Doppler HWHM constant: γ_D = _CTGAMD × ν₀ × √(T/M_amu)

# Natural isotopologue abundances (HITRAN); used to rank bands by atmospheric impact.
const _CO2_ISO_ABUND = Dict(
    1 => 0.984204,  2 => 0.011057,  3 => 0.003947,  4 => 0.000734,
    5 => 4.434e-5,  6 => 1.625e-5,  7 => 3.957e-6,  8 => 1.4717e-7,
    9 => 6.554e-8, 10 => 0.0,
)
const _CH4_ISO_ABUND = Dict(
    1 => 0.988274, 2 => 0.0111031, 3 => 0.000615751, 4 => 6.91785e-6,
)
const _LM_ISO_ABUND = Dict(2 => _CO2_ISO_ABUND, 6 => _CH4_ISO_ABUND)

# Per-species lookups. Unknown molecule ⇒ scalar default (NOT another species'
# table); unknown isotopologue within a known molecule ⇒ scalar default.
_lm_mass_amu(molecule, iso)  = get(get(_LM_MASS_AMU,  Int(molecule), Dict{Int,Float64}()), Int(iso), 44.0)
_lm_iso_abund(molecule, iso) = get(get(_LM_ISO_ABUND, Int(molecule), Dict{Int,Float64}()), Int(iso), 1e-8)

"""
    default_vpw_whitelist(data; n_top=5, isotopes=nothing, ν_window=nothing) -> Set{String}

Build a default VP_W band whitelist.

- `isotopes === nothing`: return the `n_top` bands with the highest integrated
  band intensity `S(T_REF) × natural isotopologue abundance` across all
  isotopologues (the original behaviour).
- `isotopes` given (e.g. `[1, 2]`): return the union of the top-`n_top` bands
  *per isotope* by raw `S(T_REF)` (abundance weighting drops out since the
  per-isotope sets are disjoint).
- `ν_window=(ν_lo, ν_hi)`: restrict scoring to bands whose [ν_min, ν_max]
  overlaps the window.  Useful when `data` was loaded over a wider range than
  the forward model will evaluate, so the whitelist stays relevant to the
  modelled region.

Used as the default whitelist by [`VPWLineMixing`](@ref).
"""
# Integrated band intensity at T_REF (cm/molec). Used to drop negligible bands from
# the LM sum (#4 cutoff). The S-file DipoT/PopuT0 ALREADY include isotopologue
# abundance (verified: ΣS_T matches the abundance-scaled HITRAN .par intensity to
# ~10% for both iso-1 and iso-2), so this is the true atmospheric band intensity —
# do NOT multiply by abundance again (that double-counts and mis-ranks minor isotopes).
# Layer-independent (T_REF fixed), so per-band LM contributions scale with it.
# Integrated band intensity at T_REF, used by the #4 `min_band_strength` cutoff to
# decide which coupled bands are too weak to bother evaluating. Threshold guidance
# (see scripts/sweep_band_cutoff.jl): 1e-25 → ~1.4× end-to-end at ≤0.6 mK (drop ~50%),
# 1e-26 → ~1.1-1.2× at ≤0.09 mK. Default cutoff is 0.0 (exact, all bands kept).
# NOTE: the S-file DipoT/PopuT0 ALREADY include isotopologue abundance — do NOT
# multiply by abundance again here (that double-count made minor-iso bands look ~1/abund
# too weak and wrongly pruned them, causing 2+ K BT errors).
function _band_eff_strength(band::RelmatBand)::Float64
    S = 0.0
    for line in band.lines
        stim = 1.0 - exp(-_CT_LM * line.ν / _T0_LM)
        S += line.DipoT^2 * line.PopuT0 * line.ν * stim
    end
    return S
end

function default_vpw_whitelist(data::HITRANRelmatData;
                               n_top::Int = 5,
                               isotopes::Union{Nothing, Vector{Int}} = nothing,
                               ν_window::Union{Nothing, Tuple{Real,Real}} = nothing)::Set{String}
    band_S = _band_eff_strength

    in_window(band) = isnothing(ν_window) ||
                      (band.ν_max >= ν_window[1] && band.ν_min <= ν_window[2])

    if isnothing(isotopes)
        scored = Tuple{String, Float64}[]
        for band in data.bands
            in_window(band) || continue
            abund = _lm_iso_abund(band.molecule, band.isot)
            push!(scored, (band.name, band_S(band) * abund))
        end
        sort!(scored, by=x->x[2], rev=true)
        n = min(n_top, length(scored))
        return Set(p[1] for p in @view scored[1:n])
    end

    result = Set{String}()
    for iso in isotopes
        scored = Tuple{String, Float64}[]
        for band in data.bands
            Int(band.isot) == iso || continue
            in_window(band) || continue
            push!(scored, (band.name, band_S(band)))
        end
        sort!(scored, by=x->x[2], rev=true)
        n = min(n_top, length(scored))
        for p in @view scored[1:n]
            push!(result, p[1])
        end
    end
    return result
end

# ── Full-matrix (VP_W) eigenmode representation ──────────────────────────────
"""
    BandModes

Eigenmode representation of a coupled CO₂ band at fixed (T, p).  The pole-sum
cross section is

    σ(ν) = (f/√π) · Re Σₖ conj(Aₖ) · w(zₖ),
    zₖ = (ν − Re λₖ) · f + i · Im(λₖ) · f.

Convention: λₖ has positive imaginary part (upper half plane).  In the
no-coupling limit Aₖ → Sₖ(T), λₖ → ν₀ₖ + i·p·γ_L,k, recovering Voigt.

# Fields
- `poles`:      λₖ = perturbed complex line centres (cm⁻¹), `Im(λₖ) > 0`.
- `amplitudes`: Aₖ = (Σᵢ D̂ᵢ Xᵢₖ)·(Σⱼ (X⁻¹)ₖⱼ ρⱼ D̂ⱼ), with D̂ᵢ² ρᵢ = Sᵢ(T).
- `f`:          √ln2 / γ_D evaluated at the band-mean line centre (band-uniform).
"""
struct BandModes
    poles::Vector{ComplexF64}
    amplitudes::Vector{ComplexF64}
    f::Float64
end

"""
    band_modes(band, wtfit, T, p_atm; keep_threshold=0.0) -> BandModes

Diagonalise the LM matrix M = D + i·p·W for a CO₂ band at temperature T and
pressure `p_atm` (atm), returning poles λₖ and intensity-folded amplitudes Aₖ.
The Doppler scale `f` is evaluated at the band-mean line centre (per-band, not
per-pole — mixing scrambles the line ↔ pole correspondence, but γ_D varies
only ~3% across a typical 20 cm⁻¹ band).

D̂ᵢ = DipoTᵢ · √(νᵢ · stim(T)) so D̂² ρ = S(T); see [`BandModes`](@ref).

Poles with `Im(λ) ≤ 0` are always dropped (numerical artifacts that overflow
the Faddeeva function).  If `keep_threshold > 0`, poles with
`|Aₖ| < keep_threshold × max(|Aₖ|)` are also dropped — the |A| distribution
across a band typically spans many orders of magnitude, so threshold ~1e-4
cuts ~40% of poles with no visible BT impact.
"""
function band_modes(band::RelmatBand, wtfit::W0B0Table, T::Float64,
                    p_atm::Float64;
                    keep_threshold::Float64 = 0.0,
                    Wρ::Union{Nothing, Tuple{Matrix{Float64}, Vector{Float64}}} = nothing)::BandModes
    n = length(band.lines)
    n == 0 && return BandModes(ComplexF64[], ComplexF64[], 0.0)

    # `Wρ` supplies a precomputed `_build_W_matrix(band, wtfit, T)` (lever-3
    # cache: the Jacobian's centre and ∂p calls share T, and W carries no p).
    W, ρ = Wρ === nothing ? _build_W_matrix(band, wtfit, T) : Wρ

    ν₀    = Float64[line.ν for line in band.lines]
    DipoT = Float64[line.DipoT for line in band.lines]

    # D̂ᵢ = DipoTᵢ · √(νᵢ · stim(T))  ⇒  D̂ᵢ² · ρᵢ = Sᵢ(T)
    d_eff = Vector{ComplexF64}(undef, n)
    for i in 1:n
        stim_T = 1.0 - exp(-_CT_LM * ν₀[i] / T)
        d_eff[i] = DipoT[i] * sqrt(ν₀[i] * stim_T)
    end

    # M = diag(ν₀) + i·p·W  (complex, non-Hermitian, eigenvalues in upper half plane)
    M = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        if i == j
            M[i, j] = complex(ν₀[i], p_atm * W[i, j])
        else
            M[i, j] = complex(0.0, p_atm * W[i, j])
        end
    end

    F = eigen(M)
    λ = F.values
    X = F.vectors

    # aₖ = Σᵢ D̂ᵢ Xᵢₖ;  bₖ = Σⱼ (X⁻¹)ₖⱼ ρⱼ D̂ⱼ;  Aₖ = aₖ · bₖ
    # (X \ · via the LU factorization — cheaper and better conditioned than inv(X))
    a = transpose(X) * d_eff
    b = X \ (ρ .* d_eff)
    A = a .* b

    iso      = Int(band.isot == 10 ? 10 : band.isot)
    M_amu    = _lm_mass_amu(band.molecule, iso)
    ν_center = sum(ν₀) / n
    γ_D      = _CTGAMD * ν_center * sqrt(T / M_amu)
    f        = _SQRT_LN2 / max(γ_D, 1e-10)

    # Drop poles with Im(λ) ≤ 0: numerical artifacts of the complex
    # eigendecomposition of a weakly non-Hermitian matrix (M is almost-real
    # plus a small ip·W perturbation). They carry vanishingly small |Aₖ|
    # (typically 1e-30 or below) but would make w(zₖ) overflow in the
    # evaluator since exp(−z²) diverges for Im z < 0.
    keep_mask = imag.(λ) .> 0
    if keep_threshold > 0 && any(keep_mask)
        max_A = maximum(abs.(A[keep_mask]))
        keep_mask .&= abs.(A) .>= (keep_threshold * max_A)
    end
    if !all(keep_mask)
        λ = λ[keep_mask]
        A = A[keep_mask]
    end

    return BandModes(λ, A, f)
end

"""
    compute_vpw_band_xsec(ν_grid, modes; cutoff=25.0) -> Vector{Float64}

Pole-sum cross section (cm²/molec) for a single coupled band:

    σ(ν) = (f/√π) · Re Σₖ conj(Aₖ) · w(zₖ),
    zₖ = (ν − Re λₖ)·f + i·Im(λₖ)·f.

Only poles with `|ν − Re λₖ| ≤ cutoff` contribute (matching the MT-CKD/HITRAN
wing convention used by `compute_voigt_cross_sections`).  Threaded over ν.

Uses the identity `w(x + iy) = erfcx(y − ix)` so the inner loop calls
`erfcx(complex(y, -x))` once per (ν, pole) — same primitive and same cost as
the VP_Y dispersive evaluator.
"""
function compute_vpw_band_xsec(ν_grid::WavenumberGrid, modes::BandModes;
                                cutoff::Float64 = 25.0,
                                x_far::Float64 = _X_FAR)::Vector{Float64}
    n_ν = ν_grid.n
    σ   = zeros(Float64, n_ν)

    n_p = length(modes.poles)
    n_p == 0 && return σ

    f      = modes.f
    re_λ   = real.(modes.poles)
    im_λ   = imag.(modes.poles)
    A_re   = real.(modes.amplitudes)
    A_im   = imag.(modes.amplitudes)
    prefac = f * _INV_SQRT_PI

    # Band ν-window: poles only contribute within ±cutoff of their Re position, so
    # restrict the (threaded) ν loop to that index span instead of the whole grid.
    # The per-pole cutoff check below still guarantees correctness for any margin.
    i_lo = searchsortedfirst(ν_grid.ν, minimum(re_λ) - cutoff)
    i_hi = searchsortedlast(ν_grid.ν,  maximum(re_λ) + cutoff)
    i_lo > i_hi && return σ

    Threads.@threads for i in i_lo:i_hi
        νi  = ν_grid.ν[i]
        acc = 0.0
        for k in 1:n_p
            Δν = νi - re_λ[k]
            abs(Δν) > cutoff && continue
            x = Δν * f
            y = im_λ[k] * f
            # Far-wing analytic limit: beyond |x|>x_far the Gaussian core has
            # decayed and w(x+iy) → (y + i·x)/(√π(x²+y²)) — the same asymptotics
            # the main Voigt path uses (lossless to ~3e-5 at x_far=122). Skips erfcx.
            if abs(x) > x_far
                denom = _INV_SQRT_PI / (x*x + y*y)
                rew = y * denom
                imw = x * denom
            else
                wz = erfcx(complex(y, -x))         # w(x + iy)
                rew = real(wz); imw = imag(wz)
            end
            # Re[conj(Aₖ) · w] = A_re · Re w + A_im · Im w
            acc += A_re[k] * rew + A_im[k] * imw
        end
        σ[i] = acc * prefac
    end
    return σ
end

"""
    _diagonal_band_modes(band, T, p_atm, f) -> BandModes

Construct the no-coupling (diagonal-W) `BandModes` for a band analytically:
poles = ν₀ + i·p·γ_L per line, amplitudes = S(T) real per line.  Reuses the
band-uniform Doppler factor `f` from the full-coupling call so the perturbation
`σ_full − σ_baseline` is exact in the no-coupling limit.
"""
function _diagonal_band_modes(band::RelmatBand, T::Float64, p_atm::Float64,
                                f::Float64)::BandModes
    n = length(band.lines)
    poles      = Vector{ComplexF64}(undef, n)
    amplitudes = Vector{ComplexF64}(undef, n)

    iso       = Int(band.isot == 10 ? 10 : band.isot)
    RatioPart = partition_function(Int(band.molecule), iso, T_REF) / partition_function(Int(band.molecule), iso, T)

    for (i, line) in enumerate(band.lines)
        γ_L = Float64(line.gV_air) * (_T0_LM / T)^Float64(line.n_air) * p_atm
        poles[i] = complex(Float64(line.ν), γ_L)

        PopuT  = line.PopuT0 * RatioPart *
                 exp(-_CT_LM * line.E_lower * (1.0/T - 1.0/_T0_LM))
        stim_T = 1.0 - exp(-_CT_LM * line.ν / T)
        S_T    = line.DipoT^2 * PopuT * line.ν * stim_T
        amplitudes[i] = complex(S_T, 0.0)
    end
    return BandModes(poles, amplitudes, f)
end

"""
    compute_vpw_band_perturbation(ν_grid, band, wtfit, T, p_atm; cutoff=25.0)

LM-only perturbation for a single band:

    Δσ_b(ν) = σ_vpw_b(ν) − σ_vpw_b_no_coupling(ν)

i.e. the full eigendecomposed cross section minus the same band evaluated
with the off-diagonal couplings forced to zero (using the same band-uniform
γ_D and S-file parameters).  In the no-coupling limit Δσ_b → 0 exactly,
which lets the perturbation slot into a HITRAN-Voigt baseline without
double-counting that band's lines.
"""
function compute_vpw_band_perturbation(ν_grid::WavenumberGrid, band::RelmatBand,
                                        wtfit::W0B0Table, T::Float64, p_atm::Float64;
                                        cutoff::Float64 = 25.0,
                                        keep_threshold::Float64 = 1e-4,
                                        x_far::Float64 = _X_FAR,
                                        Wρ::Union{Nothing, Tuple{Matrix{Float64}, Vector{Float64}}} = nothing)::Vector{Float64}
    modes_full = band_modes(band, wtfit, T, p_atm; keep_threshold=keep_threshold, Wρ=Wρ)
    σ_full     = compute_vpw_band_xsec(ν_grid, modes_full; cutoff=cutoff, x_far=x_far)
    modes_base = _diagonal_band_modes(band, T, p_atm, modes_full.f)
    σ_base     = compute_vpw_band_xsec(ν_grid, modes_base; cutoff=cutoff, x_far=x_far)
    return σ_full .- σ_base
end

"""
    compute_voigt_vpw_cross_sections(ν_grid, ll_co2, relmat, T, p_atm;
                                      whitelist, cutoff=25.0) -> Vector{Float64}

Hybrid CO₂ cross section: full HITRAN Voigt baseline + VP_W perturbation for
bands in `whitelist`, + VP_Y dispersive perturbation for the rest.

Algorithm per band:
- `band ∈ whitelist` → `compute_vpw_band_perturbation` (eigendecomp + pole sum,
  baselined against the diagonal-W limit so the LM-only contribution is added).
- `band ∉ whitelist` → `_lm_band_dispersive` (the per-band VP_Y kernel that
  `compute_lm_dispersive_correction` shares).

The total cross section is `max(σ_voigt + Δσ_total, 0)`.
"""
function compute_voigt_vpw_cross_sections(ν_grid::WavenumberGrid,
                                            ll_co2::HITRANLinelist,
                                            relmat::HITRANRelmatData,
                                            T::Float64, p_atm::Float64;
                                            whitelist::Set{String},
                                            cutoff::Float64 = 25.0,
                                            lm_cutoff::Float64 = cutoff,
                                            keep_threshold::Float64 = 1e-4,
                                            min_band_strength::Float64 = 0.0,
                                            x_far::Float64 = _X_FAR)::Vector{Float64}
    σ_voigt = compute_voigt_cross_sections(ν_grid, ll_co2, T, p_atm; cutoff=cutoff)
    Δσ = _vpw_perturbation(ν_grid, relmat, T, p_atm; whitelist=whitelist, cutoff=lm_cutoff,
                           keep_threshold=keep_threshold, min_band_strength=min_band_strength,
                           x_far=x_far)
    return max.(σ_voigt .+ Δσ, 0.0)
end

"""
    _vpw_perturbation(ν_grid, relmat, T, p_atm; whitelist, cutoff, keep_threshold,
                      min_band_strength, x_far) -> Vector{Float64}

VP_W line-mixing perturbation Δσ (cm²/molec), summed over bands: whitelisted bands
use the eigendecomposed `compute_vpw_band_perturbation`, the rest the VP_Y
`_lm_band_dispersive`. This is the LM-only additive term (no Voigt baseline, no
`max(·,0)` clamp) — the perturbation the Jacobian central-differences in `(T, p)`.
"""
function _vpw_perturbation(ν_grid::WavenumberGrid, relmat::HITRANRelmatData,
                           T::Float64, p_atm::Float64;
                           whitelist::Set{String},
                           cutoff::Float64 = 25.0,
                           keep_threshold::Float64 = 1e-4,
                           min_band_strength::Float64 = 0.0,
                           x_far::Float64 = _X_FAR,
                           Ys = nothing,
                           Wρs::Union{Nothing, Dict{Int, Tuple{Matrix{Float64}, Vector{Float64}}}} = nothing)::Vector{Float64}
    # Non-whitelisted bands: one flat dispersive launch over all of them.
    Δσ = compute_lm_dispersive_correction(ν_grid, relmat, T, p_atm;
                                          cutoff=cutoff, x_far=x_far,
                                          min_band_strength=min_band_strength,
                                          skip=whitelist, Ys=Ys)
    # Whitelisted bands: eigendecomposed per-band perturbation (a handful).
    for (bi, band) in enumerate(relmat.bands)
        band.name in whitelist || continue
        Int(band.li) > 8 && continue
        # #4 cutoff: skip bands whose abundance-weighted intensity is below threshold.
        min_band_strength > 0.0 && _band_eff_strength(band) < min_band_strength && continue
        lli = Int8(min(band.li, band.lf))
        llf = Int8(max(band.li, band.lf))
        wtfit = get(relmat.wtfit, (lli, llf), nothing)
        wtfit === nothing && continue
        Wρ = Wρs === nothing ? nothing : get(Wρs, bi, nothing)
        Δσ .+= compute_vpw_band_perturbation(ν_grid, band, wtfit, T, p_atm;
                                               cutoff=cutoff,
                                               keep_threshold=keep_threshold,
                                               x_far=x_far, Wρ=Wρ)
    end
    return Δσ
end

# ── Cross-section computation ─────────────────────────────────────────────────

# Single-launch dispersive evaluator (lever 5). All active lines — every band,
# compact (no placeholder slots) — are concatenated, sorted by (shifted) centre,
# and evaluated in ONE windowed KernelAbstractions kernel, mirroring
# `voigt_cross_section_kernel!`: one work-item per grid point, binary-search
# windowing on the sorted centres, in-place accumulation. This replaces the old
# per-band path (a full-grid `zeros(n_ν)` allocation + `Threads.@threads` launch
# per band — ~170 MB of transient allocation and 339 launches per call at
# production size). CPU-only: the body needs the complex `erfcx`.
@kernel function lm_dispersive_kernel!(Δσ, ν_grid, lν0, lf, ly, lYSn, cutoff, x_far)
    i   = @index(Global, Linear)
    ν   = ν_grid[i]
    n_L = length(lν0)

    # Lines sorted by ν0; window to [ν-cutoff, ν+cutoff] (⇒ |Δν| ≤ cutoff).
    j_lo = _lower_bound(lν0, ν - cutoff, n_L)
    j_hi = _upper_bound(lν0, ν + cutoff, n_L)

    acc = 0.0
    @inbounds for j in j_lo:j_hi
        x  = (ν - lν0[j]) * lf[j]
        yk = ly[j]
        # Far-wing analytic limit of Im[w]: x/(√π(x²+y²)) beyond |x|>x_far
        # (Gaussian core decayed). Same shortcut as the main Voigt path; skips erfcx.
        imw = abs(x) > x_far ? x * _INV_SQRT_PI / (x*x + yk*yk) :
                               imag(erfcx(complex(yk, -x)))
        acc += lYSn[j] * imw
    end
    Δσ[i] = acc
end

# Append the active dispersive lines of one band (Y ≠ 0 and S_T > 0) to the flat
# per-line arrays: shifted centre ν0, Doppler factor f, y = γ_L·f, and the
# per-line weight YSnorm = Y·p·S(T)·f/√π. Same per-line math as the old
# `_lm_band_dispersive` prologue, but compact — inactive lines are simply not
# appended (no uninitialized placeholder slots).
function _append_dispersive_lines!(ν0s::Vector{Float64}, fs::Vector{Float64},
                                   ys::Vector{Float64}, YSns::Vector{Float64},
                                   band::RelmatBand, Y_band::Vector{Float64},
                                   T::Float64, p_atm::Float64)
    M_amu        = _lm_mass_amu(band.molecule, Int(band.isot))
    iso          = Int(band.isot == 10 ? 10 : band.isot)
    Q_ratio_band = partition_function(Int(band.molecule), iso, T_REF) / partition_function(Int(band.molecule), iso, T)

    for (k, rl) in enumerate(band.lines)
        Y_band[k] == 0.0 && continue     # zero weight either way
        ν0  = Float64(rl.ν) + Float64(rl.shift) * p_atm
        γ_D = _CTGAMD * rl.ν * sqrt(T / M_amu)
        γ_L = Float64(rl.gV_air) * (_T0_LM / T)^Float64(rl.n_air) * p_atm
        f   = _SQRT_LN2 / max(γ_D, 1e-10)

        stim_T0 = 1.0 - exp(-_CT_LM * rl.ν / _T0_LM)
        stim_T  = 1.0 - exp(-_CT_LM * rl.ν / T)
        S0  = rl.DipoT^2 * rl.PopuT0 * rl.ν * stim_T0
        S_T = S0 * Q_ratio_band *
              exp(-_CT_LM * rl.E_lower * (1.0/T - 1.0/_T0_LM)) *
              stim_T / stim_T0
        S_T <= 0.0 && continue

        push!(ν0s, ν0); push!(fs, f); push!(ys, γ_L * f)
        push!(YSns, Y_band[k] * p_atm * S_T * f * _INV_SQRT_PI)
    end
    return nothing
end

# Sort the flat line arrays by centre and run the single kernel launch.
function _lm_dispersive_eval(ν_grid::WavenumberGrid,
                             ν0s::Vector{Float64}, fs::Vector{Float64},
                             ys::Vector{Float64}, YSns::Vector{Float64};
                             cutoff::Float64, x_far::Float64)::Vector{Float64}
    Δσ = zeros(Float64, ν_grid.n)
    isempty(ν0s) && return Δσ
    ord = sortperm(ν0s)
    backend = CPU()
    kernel! = lm_dispersive_kernel!(backend, 256)
    kernel!(Δσ, ν_grid.ν, ν0s[ord], fs[ord], ys[ord], YSns[ord], cutoff, x_far;
            ndrange = ν_grid.n)
    KernelAbstractions.synchronize(backend)
    return Δσ
end

# Per-band VP_Y dispersive perturbation (single-band flat evaluation). Kept for
# tests and fine-grained use; the production paths batch all bands into one
# launch via `compute_lm_dispersive_correction`. `Y` overrides the Y(T) build
# (lever 3 cache).
function _lm_band_dispersive(ν_grid::WavenumberGrid, band::RelmatBand,
                              wtfit::W0B0Table, T::Float64, p_atm::Float64;
                              cutoff::Float64 = 25.0,
                              x_far::Float64 = _X_FAR,
                              Y::Union{Nothing, Vector{Float64}} = nothing)::Vector{Float64}
    Y_band = Y === nothing ? _calc_W_and_Y(band, wtfit, T) : Y
    ν0s = Float64[]; fs = Float64[]; ys = Float64[]; YSns = Float64[]
    _append_dispersive_lines!(ν0s, fs, ys, YSns, band, Y_band, T, p_atm)
    return _lm_dispersive_eval(ν_grid, ν0s, fs, ys, YSns; cutoff=cutoff, x_far=x_far)
end

"""
    _lm_dispersive_Ys(relmat, T; min_band_strength=0.0, skip=nothing)
        -> Vector{Union{Nothing, Vector{Float64}}}

Y(T) for every band that passes the dispersive-path filters (aligned with
`relmat.bands`; `nothing` for filtered/skipped bands). The lever-3 cache: the
Jacobian's centre and both ∂p perturbation calls share one temperature, so the
O(n²) Y build runs once instead of three times per layer.
"""
function _lm_dispersive_Ys(relmat::HITRANRelmatData, T::Float64;
                           min_band_strength::Float64 = 0.0,
                           skip::Union{Nothing, Set{String}} = nothing)
    Ys = Vector{Union{Nothing, Vector{Float64}}}(nothing, length(relmat.bands))
    for (bi, band) in enumerate(relmat.bands)
        Int(band.li) > 8 && continue
        min_band_strength > 0.0 && _band_eff_strength(band) < min_band_strength && continue
        wtfit = get(relmat.wtfit, (Int8(min(band.li, band.lf)), Int8(max(band.li, band.lf))), nothing)
        wtfit === nothing && continue
        skip !== nothing && band.name in skip && continue
        Ys[bi] = _calc_W_and_Y(band, wtfit, T)
    end
    return Ys
end

"""
    compute_lm_dispersive_correction(ν_grid, relmat, T, p_atm;
                                      cutoff=25.0) -> Vector{Float64}

Dispersive line-mixing perturbation (cm²/molec) for CO2:

    σ_disp(ν) = Σ_bands Σᵢ Yᵢ(T,p) · Sᵢ(T) · (f/√π) · Im[w(zᵢ)]

where the sum runs over S-file LM lines.  The Fortran validity clip
|Yᵢ × p| > 0.08 is intentionally NOT applied — ARTS evaluates the formula
for all lines and the W-matrix sum rule keeps the band-integrated effect
near zero.  Clipping low-J Q lines (which violate first-order Rosenkranz
validity at p > ~0.5 atm) removes the dominant dispersive correction in
the Q-branch wings and produces large BT residuals.  Per-line wing
artifacts at sub-cm⁻¹ scales are washed out by the IASI ILS / output grid.

This is a *perturbation* — the returned values can be locally negative.
Add to a Voigt baseline (`compute_voigt_cross_sections` with the full HITRAN
CO2 linelist) to obtain the total cross section.

Line parameters (ν₀, γ_L, γ_D, S(T)) come from the S-files, consistent with
the CalcW Y computation.  All bands' active lines are evaluated in a single
windowed kernel launch (`lm_dispersive_kernel!`).  Internal kwargs: `skip`
excludes named bands (the VP_W whitelist), `Ys` supplies precomputed Y(T)
per band (`_lm_dispersive_Ys`, lever-3 cache).
"""
function compute_lm_dispersive_correction(ν_grid::WavenumberGrid,
                                           relmat::HITRANRelmatData,
                                           T::Float64,
                                           p_atm::Float64;
                                           cutoff::Float64 = 25.0,
                                           min_band_strength::Float64 = 0.0,
                                           x_far::Float64 = _X_FAR,
                                           skip::Union{Nothing, Set{String}} = nothing,
                                           Ys = nothing)::Vector{Float64}
    ν0s = Float64[]; fs = Float64[]; ys = Float64[]; YSns = Float64[]
    for (bi, band) in enumerate(relmat.bands)
        Int(band.li) > 8 && continue
        # #4 cutoff: skip bands whose abundance-weighted intensity is below threshold.
        min_band_strength > 0.0 && _band_eff_strength(band) < min_band_strength && continue
        lli = Int8(min(band.li, band.lf))
        llf = Int8(max(band.li, band.lf))
        wtfit = get(relmat.wtfit, (lli, llf), nothing)
        wtfit === nothing && continue
        skip !== nothing && band.name in skip && continue
        Y_band = if Ys === nothing || Ys[bi] === nothing
            _calc_W_and_Y(band, wtfit, T)
        else
            Ys[bi]::Vector{Float64}
        end
        _append_dispersive_lines!(ν0s, fs, ys, YSns, band, Y_band, T, p_atm)
    end
    return _lm_dispersive_eval(ν_grid, ν0s, fs, ys, YSns; cutoff=cutoff, x_far=x_far)
end

"""
    compute_voigt_lm_cross_sections(ν_grid, ll_co2, relmat, T, p_atm;
                                     cutoff=25.0,
                                     method=FullFaddeeva) -> Vector{Float64}

CO2 absorption cross-section (cm²/molec) including first-order Rosenkranz
line mixing.  Convenience wrapper:

    σ(ν) = max(σ_voigt(ν) + σ_disp(ν), 0)

with `σ_voigt` from `compute_voigt_cross_sections(ν_grid, ll_co2, ...)` and
`σ_disp` from `compute_lm_dispersive_correction(ν_grid, relmat, ...)`.

For finer control (e.g. composing with custom Voigt parameters), call the two
underlying functions separately.
"""
function compute_voigt_lm_cross_sections(ν_grid::WavenumberGrid,
                                          ll_co2::HITRANLinelist,
                                          relmat::HITRANRelmatData,
                                          T::Float64,
                                          p_atm::Float64;
                                          cutoff::Float64 = 25.0,
                                          lm_cutoff::Float64 = cutoff,
                                          method::VoigtMethod = FullFaddeeva,
                                          min_band_strength::Float64 = 0.0,
                                          x_far::Float64 = _X_FAR)::Vector{Float64}
    σ_voigt = compute_voigt_cross_sections(ν_grid, ll_co2, T, p_atm;
                                            cutoff=cutoff, method=method, x_far=x_far)
    σ_disp  = compute_lm_dispersive_correction(ν_grid, relmat, T, p_atm;
                                                cutoff=lm_cutoff, x_far=x_far,
                                                min_band_strength=min_band_strength)
    return max.(σ_voigt .+ σ_disp, 0.0)
end

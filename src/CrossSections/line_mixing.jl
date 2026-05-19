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

struct RelmatBand
    name::String
    li::Int8          # vibrational angular momentum, lower state
    lf::Int8          # vibrational angular momentum, upper state
    isot::Int8
    ν_min::Float64
    ν_max::Float64
    lines::Vector{RelmatLine}
end

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
end

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
    VPYLineMixing(data::HITRANRelmatData)

First-order Rosenkranz (Voigt + dispersive perturbation) line mixing.
Pass to `iasi_forward_model(...; line_mixing=VPYLineMixing(relmat))` to enable
LM on the CO2 channel.  See `compute_voigt_lm_cross_sections`.
"""
struct VPYLineMixing <: AbstractLineMixing
    data::HITRANRelmatData
end

# Per-species cross-section dispatch used by `iasi_forward_model`.
# `nothing` and any non-CO2 species fall through to the plain Voigt path.
# `VPYLineMixing` + CO2 routes through the Voigt+dispersive wrapper.
function _species_cross_section(::Nothing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend)
    compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                  vmr_self=vmr_self, cutoff=cutoff, backend=backend)
end

function _species_cross_section(lm::AbstractLineMixing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend)
    compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                  vmr_self=vmr_self, cutoff=cutoff, backend=backend)
end

function _species_cross_section(lm::VPYLineMixing, sp::GasSpecies, ν_grid, ll, T, p_atm;
                                 vmr_self::Float64, cutoff::Float64, backend)
    sp == CO2 || return compute_voigt_cross_sections(ν_grid, ll, T, p_atm;
                                                      vmr_self=vmr_self, cutoff=cutoff, backend=backend)
    compute_voigt_lm_cross_sections(ν_grid, ll, lm.data, T, p_atm; cutoff=cutoff)
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
    end

    return HITRANRelmatData(bands, loaded_wtfit)
end

# ── Y coefficient computation (CalcW in Fortran) ─────────────────────────────

"""
    _calc_W_and_Y(band, wtfit, T) -> Vector{Float64}

Compute the first-order Rosenkranz Y coefficient (per atm) for each line in
`band` at temperature T.  Returns a vector of length n_lines.

Algorithm (CalcW in LM_calc_CO2.for):
  1. Compute W(T) = exp(W0) × (T/T0)^B0 for each off-diagonal pair.
  2. Sort lines by decreasing intensity (S ∝ DipoT² × ν × PopuT).
  3. Apply sum-rule correction to off-diagonal elements.
  4. Compute Y_i = 2 × Σⱼ≠ᵢ |DipoTⱼ|/|DipoTᵢ| × W_{j,i} / (νᵢ − νⱼ).
"""
function _calc_W_and_Y(band::RelmatBand, wtfit::W0B0Table, T::Float64)::Vector{Float64}
    n = length(band.lines)
    n == 0 && return Float64[]

    li = Int(band.li)
    lf = Int(band.lf)
    lli = min(li, lf)
    llf = max(li, lf)

    # Bypass: no LM when li > 8 or |li-lf| > 1 (Fortran skip trick)
    if li > 8 || lf > 8 || abs(li - lf) > 1
        return zeros(Float64, n)
    end

    # Temperature-scaled populations and dipoles
    # Use TIPS-2024 partition function for this band's isotopologue (mol_id=2=CO2)
    iso = Int(band.isot == 10 ? 10 : band.isot)
    RatioPart = partition_function(2, iso, T_REF) / partition_function(2, iso, T)
    PopuT = Vector{Float64}(undef, n)
    DipoT = Vector{Float64}(undef, n)
    ν_arr = Vector{Float64}(undef, n)
    for (i, line) in enumerate(band.lines)
        PopuT[i] = line.PopuT0 * RatioPart * exp(-_CT_LM * line.E_lower * (1.0/T - 1.0/_T0_LM))
        DipoT[i] = line.DipoT
        ν_arr[i] = line.ν
    end

    # Effective intensity for sorting
    S_eff = [ν_arr[i] * PopuT[i] * DipoT[i]^2 for i in 1:n]

    # Sort indices by decreasing S_eff (matches Fortran CalcW sort)
    ord = sortperm(S_eff, rev=true)
    inv_ord = invperm(ord)

    ν_s    = ν_arr[ord]
    DipoT_s= DipoT[ord]
    Dipo0_s= [band.lines[ord[i]].Dipo0 for i in 1:n]
    Ji_s   = [Int(band.lines[ord[i]].Ji) for i in 1:n]
    br_s   = [Int(band.lines[ord[i]].branch) for i in 1:n]
    PopuT_s= PopuT[ord]

    # Build W matrix (off-diagonal elements only; diagonal not needed for Y).
    # Fortran convention: WTfit stores entries for jic ≥ jipc only.
    # The Fortran code uses full nested loops but skips pairs where jjip > jji
    # (i.e., where the coupling partner has higher J than the current line).
    # This ensures we only look up keys (Ji_ir, Ji_irp) where Ji_ir ≥ Ji_irp.
    W = zeros(Float64, n, n)
    dlgT0T = log(_T0_LM / T)
    isot = Int(band.isot)

    for ir in 1:n
        jji  = Ji_s[ir];  jjf  = Ji_s[ir]  + br_s[ir]
        for irp in 1:n
            irp == ir && continue
            jjip = Ji_s[irp]; jjfp = Ji_s[irp] + br_s[irp]

            # Fortran skip: only process pairs where Ji_irp ≤ Ji_ir
            jjip > jji && continue

            # Asymmetric isotopologue symmetry restriction
            if isot > 2 && isot != 7 && isot != 10
                abs(jji - jjip) % 2 != 0 && continue
            end

            # WTfit convention: if li > lf swap J roles for table lookup
            if li <= lf
                ji_eff = jji; jf_eff = jjf; jip_eff = jjip; jfp_eff = jjfp
            else
                ji_eff = jjf; jf_eff = jji; jip_eff = jjfp; jfp_eff = jjip
            end
            key = (ji_eff, jip_eff)

            type_i  = ji_eff > jf_eff   ? :p : (ji_eff  == jf_eff  ? :q : :r)
            type_ip = jip_eff > jfp_eff ? :p : (jip_eff == jfp_eff ? :q : :r)
            sub = Symbol(type_i, type_ip)

            tbl = getfield(wtfit, sub)
            haskey(tbl, key) || continue

            W0R, B0R = tbl[key]
            ycal = exp(W0R - B0R * dlgT0T)

            # W[irp, ir]: direct lookup value
            # W[ir, irp]: upper triangle via detailed balance
            W[irp, ir] = ycal
            W[ir, irp] = ycal * PopuT_s[ir] / PopuT_s[irp]
        end
    end

    # Force all off-diagonal elements negative (energy flows out of each state)
    for ir in 1:n, irp in 1:n
        ir == irp && continue
        W[ir, irp] = -abs(W[ir, irp])
    end

    # Set diagonal to air-broadened half-width at T, P=1 atm (Fortran: w(ir,ir)=HWT(ir))
    # HWT = gV_air × (T0/T)^n_air  at 1 atm, pure air (no H2O/self correction needed
    # for the sum-rule, which only uses ratios of diagonal vs off-diagonal magnitudes)
    for ir in 1:n
        line_ir = band.lines[ord[ir]]
        W[ir, ir] = Float64(line_ir.gV_air) * ((_T0_LM / T)^Float64(line_ir.n_air))
    end

    # Sum-rule correction (Fortran: normalise lower triangle using Dipo0 ratio).
    # sumUp includes the diagonal (W[ir,ir]=HWT>0), which ensures the correction
    # factor (-sumUp/sumLW) is positive and keeps off-diagonal elements negative.
    for ir in 1:n
        sumLW = 0.0   # Σ |Dipo0_irp| × W[irp,ir] for irp > ir (below diagonal)
        sumUp = 0.0   # Σ |Dipo0_irp| × W[irp,ir] for irp ≤ ir (diagonal + above)
        for irp in 1:n
            if isot > 2 && isot != 7 && isot != 10
                abs(Ji_s[ir] - Ji_s[irp]) % 2 != 0 && continue
            end
            if irp > ir
                sumLW += abs(Dipo0_s[irp]) * W[irp, ir]
            else
                sumUp += abs(Dipo0_s[irp]) * W[irp, ir]
            end
        end

        sumLW == 0.0 && continue
        ratio = -sumUp / sumLW   # positive when diagonal dominates sumUp
        for irp in (ir+1):n
            W[irp, ir] = W[irp, ir] * ratio
            W[ir, irp] = W[irp, ir] * PopuT_s[ir] / PopuT_s[irp]
        end
    end

    # Clear diagonal before Y computation (diagonal not used in Y sum)
    for ir in 1:n; W[ir, ir] = 0.0; end

    # Compute Y_i = 2 × Σⱼ≠ᵢ |DipoTⱼ|/|DipoTᵢ| × W_{j,i} / (νᵢ − νⱼ)
    Y_sorted = zeros(Float64, n)
    for ir in 1:n
        s = 0.0
        for irp in 1:n
            irp == ir && continue
            isot = Int(band.isot)
            if isot > 2 && isot != 7 && isot != 10
                abs(Ji_s[ir] - Ji_s[irp]) % 2 != 0 && continue
            end
            Δν = ν_s[ir] - ν_s[irp]
            # Guard against degenerate ν (sign(0)=0 would collapse the regularization)
            abs(Δν) < 1e-4 && (Δν = (Δν >= 0.0 ? 1.0 : -1.0) * 1e-4)
            s += 2.0 * abs(DipoT_s[irp]) / abs(DipoT_s[ir]) * (1.0 / Δν) * W[irp, ir]
        end
        Y_sorted[ir] = s
    end

    # Invert the sort to return Y in original line order
    Y = Vector{Float64}(undef, n)
    for i in 1:n
        Y[i] = Y_sorted[inv_ord[i]]
    end
    return Y
end

# ── CO2 isotopologue molar masses (amu) for Doppler width ────────────────────

const _CO2_MASS_AMU = Dict(
    1 => 43.98983, 2 => 44.99318, 3 => 45.99398,
    4 => 44.99403, 5 => 45.99706, 6 => 44.99706,
    7 => 47.99832, 8 => 47.00134, 9 => 48.00196,
    10=> 47.00427,
)
const _CTGAMD = 3.5812e-7   # Doppler HWHM constant: γ_D = _CTGAMD × ν₀ × √(T/M_amu)

# ── Cross-section computation ─────────────────────────────────────────────────

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
the CalcW Y computation.
"""
function compute_lm_dispersive_correction(ν_grid::WavenumberGrid,
                                           relmat::HITRANRelmatData,
                                           T::Float64,
                                           p_atm::Float64;
                                           cutoff::Float64 = 25.0)::Vector{Float64}
    n_ν    = ν_grid.n
    σ_disp = zeros(Float64, n_ν)

    for band in relmat.bands
        Int(band.li) > 8 && continue

        lli = Int8(min(band.li, band.lf))
        llf = Int8(max(band.li, band.lf))
        wtfit = get(relmat.wtfit, (lli, llf), nothing)
        wtfit === nothing && continue

        Y_band  = _calc_W_and_Y(band, wtfit, T)
        n_lines = length(band.lines)

        M_amu        = get(_CO2_MASS_AMU, Int(band.isot), 44.0)
        iso          = Int(band.isot == 10 ? 10 : band.isot)
        Q_ratio_band = partition_function(2, iso, T_REF) / partition_function(2, iso, T)

        ν0_b   = Vector{Float64}(undef, n_lines)
        f_b    = Vector{Float64}(undef, n_lines)
        y_b    = Vector{Float64}(undef, n_lines)
        YSnorm = zeros(Float64, n_lines)   # Yᵢ(T,p) × S(T) × f/√π; 0 if clipped or skipped

        for (k, rl) in enumerate(band.lines)
            ν0  = Float64(rl.ν) + Float64(rl.shift) * p_atm
            γ_D = _CTGAMD * rl.ν * sqrt(T / M_amu)
            γ_L = Float64(rl.gV_air) * (_T0_LM / T)^Float64(rl.n_air) * p_atm
            f   = _SQRT_LN2 / max(γ_D, 1e-10)
            y   = γ_L * f

            stim_T0 = 1.0 - exp(-_CT_LM * rl.ν / _T0_LM)
            stim_T  = 1.0 - exp(-_CT_LM * rl.ν / T)
            S0  = rl.DipoT^2 * rl.PopuT0 * rl.ν * stim_T0
            S_T = S0 * Q_ratio_band *
                  exp(-_CT_LM * rl.E_lower * (1.0/T - 1.0/_T0_LM)) *
                  stim_T / stim_T0
            S_T <= 0.0 && continue

            ν0_b[k]  = ν0
            f_b[k]   = f
            y_b[k]   = y
            YSnorm[k] = Y_band[k] * p_atm * S_T * f * _INV_SQRT_PI
        end

        Threads.@threads for i in 1:n_ν
            νi  = ν_grid.ν[i]
            acc = 0.0
            for k in 1:n_lines
                YSnorm[k] == 0.0 && continue
                abs(νi - ν0_b[k]) > cutoff && continue
                x  = (νi - ν0_b[k]) * f_b[k]
                wz = erfcx(complex(y_b[k], -x))
                acc += YSnorm[k] * imag(wz)
            end
            σ_disp[i] += acc
        end
    end

    return σ_disp   # NOT clamped — perturbation can be locally negative
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
                                          method::VoigtMethod = FullFaddeeva)::Vector{Float64}
    σ_voigt = compute_voigt_cross_sections(ν_grid, ll_co2, T, p_atm;
                                            cutoff=cutoff, method=method)
    σ_disp  = compute_lm_dispersive_correction(ν_grid, relmat, T, p_atm;
                                                cutoff=cutoff)
    return max.(σ_voigt .+ σ_disp, 0.0)
end

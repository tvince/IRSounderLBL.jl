"""
IASI Level-1C reader (EPS Native `.nat` format, v5 — distributed since ~2010).

Reads the EUMETSAT/EPS-native IASI L1C product as delivered by NOAA CLASS: one
file ≈ one orbit. Produces measured radiance spectra (or brightness temperatures)
on the fixed IASI channel grid, the per-FOV geolocation/viewing geometry, and —
importantly for cloud screening — the **AVHRR-derived cloud fraction** collocated
in each IASI field of view.

The binary record layout (Generic Record Header framing, MDR field byte offsets,
GIADR scale factors, radiance de-scaling) is ported from Anu Dudhia's reference
Python reader `read_iasi_l1c.py` (Oxford EODG,
<https://eodg.atm.ox.ac.uk/user/dudhia/iasi/read_iasi_l1c/>). Offsets are quoted
against that reader; on the first real granule the sanity gates in
[`read_iasi_l1c`] (lat/lon ranges, BT in 150–350 K) will flag any layout drift.

Radiances are returned in the package convention **mW/(m²·sr·cm⁻¹)** — the same
units `iasi_forward_model` emits — so a measured spectrum is directly comparable
to a simulated one and feeds `optimal_estimation` without conversion.
"""

# ── EPS-native layout constants (Dudhia read_iasi_l1c.py) ──────────────────────
const _EPS_GRH_SIZE   = 20            # Generic Record Header, bytes
const _EPS_CLASS_MPHR = 1             # Main Product Header (ASCII)
const _EPS_CLASS_GIADR = 5            # Global Internal Auxiliary Data Record
const _EPS_CLASS_MDR  = 8             # Measurement Data Record
const _EPS_MDR_SUBCLASS = 2           # IASI L1C science MDR

const _IASI_NSTP  = 30                # scan steps (EFOVs) per MDR scan line
const _IASI_NPIX  = 4                 # pixels per EFOV (2×2)
const _IASI_NSPEC = 8700              # int16 samples stored per spectrum
const _IASI_NBND  = 3                 # quality-flag bands
const _IASI_WNO1  = 645.0             # first channel wavenumber (cm⁻¹)
const _IASI_WNOD  = 0.25              # channel spacing (cm⁻¹)
const _IASI_NWNO  = 8461              # valid channels (645 … 2760 cm⁻¹)

# Absolute byte offsets within an MDR record (from record start, incl. GRH).
const _OFF_QUAL    = 255260           # GQisFlagQual      [NSTP,NPIX,NBND] int8
const _OFF_GEOLOC  = 255893           # GGeoSondLoc       [NSTP,NPIX,2]    >i4 ×1e-6° (lon,lat)
const _OFF_ANG_SAT = 256853           # GGeoSondAnglesMETOP [.,.,2]        >i4 ×1e-6° (zen,azi)
const _OFF_ANG_SUN = 263813           # GGeoSondAnglesSun [NSTP,NPIX,2]    >i4 ×1e-6° (zen,azi)
const _OFF_NSFIRST = 276782           # IDefNsfirst1b                       >i4
const _OFF_NSLAST  = 276786           # IDefNslast1b                        >i4
const _OFF_SPEC    = 276790           # GS1cSpect         [NSTP,NPIX,NSPEC] >i2
const _OFF_CLDFRAC = 2728548          # GEUMAvhrr1BCldFrac  [NSTP,NPIX]     int8 (%)
const _OFF_LNDFRAC = 2728668          # GEUMAvhrr1BLandFrac [NSTP,NPIX]     int8 (%)
const _OFF_AVHQUAL = 2728788          # GEUMAvhrr1BQual     [NSTP,NPIX]     int8

# De-scaled radiance is W/(m²·sr·m⁻¹); ×1e5 → mW/(m²·sr·cm⁻¹) (package units).
const _IASI_RADFAC = 1.0e5

# ── Big-endian scalar reads at a 0-based byte offset into a UInt8 buffer ───────
@inline _be_u4(b, o) = ntoh(reinterpret(UInt32, @view b[o+1:o+4])[1])
@inline _be_i4(b, o) = ntoh(reinterpret(Int32,  @view b[o+1:o+4])[1])
@inline _be_i2(b, o) = ntoh(reinterpret(Int16,  @view b[o+1:o+2])[1])
@inline _u1(b, o)    = b[o+1]

# ── Data model ─────────────────────────────────────────────────────────────────

"""
    IASIL1CGranule

A selection of IASI L1C fields of view (FOVs) read from one `.nat` granule.
All per-FOV vectors share the index `1:nfov(granule)`; the spectral matrix `spc`
is `nwno × nfov`. Build with [`read_iasi_l1c`].

# Fields
- `file`     : source path
- `mph`      : Main Product Header (`Dict{String,String}`)
- `wno`      : channel wavenumbers (cm⁻¹), length `nwno`
- `spc`      : `nwno × nfov` radiance [mW/(m²·sr·cm⁻¹)] (or brightness temp [K] if `is_bt`)
- `is_bt`    : `true` if `spc` holds brightness temperatures
- `lat,lon`  : geolocation (degrees)
- `zen`      : satellite zenith angle θᵢ (deg)
- `azi`      : satellite azimuth angle ϕᵢ (deg)
- `sza`      : solar zenith angle θₛ (deg)
- `saa`      : solar azimuth angle ϕₛ (deg)
- `sra`      : **solar reflection (sun-glint) angle** φᵣ (deg) — the angle between
               the IASI view direction and the specular solar-reflection path off a
               flat surface (Vincent thesis Eq. 2.4). Small `sra` ⇒ IASI is looking
               into the sun glint; screen with `sralim` to avoid contaminated FOVs.
- `cld`      : AVHRR cloud fraction in the IASI FOV (%)
- `lnd`      : AVHRR land fraction (%)
- `line`     : scan-line / MDR index (1-based)
- `step`     : scan step within the MDR (1…30)
- `pix`      : pixel within the EFOV (1…4)
"""
struct IASIL1CGranule
    file::String
    mph::Dict{String,String}
    wno::Vector{Float64}
    spc::Matrix{Float64}
    is_bt::Bool
    lat::Vector{Float64}
    lon::Vector{Float64}
    zen::Vector{Float64}
    azi::Vector{Float64}
    sza::Vector{Float64}
    saa::Vector{Float64}
    sra::Vector{Float64}
    cld::Vector{Float64}
    lnd::Vector{Float64}
    line::Vector{Int}
    step::Vector{Int}
    pix::Vector{Int}
end

"""
    nfov(granule) -> Int

Number of fields of view retained in the granule.
"""
nfov(g::IASIL1CGranule) = length(g.lat)

Base.show(io::IO, g::IASIL1CGranule) =
    print(io, "IASIL1CGranule($(nfov(g)) FOVs, $(length(g.wno)) channels ",
              "$(round(first(g.wno);digits=2))–$(round(last(g.wno);digits=2)) cm⁻¹, ",
              g.is_bt ? "BT[K]" : "radiance", ")")

# ── MPHR (ASCII key = value) ─────────────────────────────────────────────────

function _parse_mphr(bytes::Vector{UInt8})::Dict{String,String}
    mph = Dict{String,String}()
    for line in split(String(bytes), '\n')
        isempty(strip(line)) && continue
        eq = findfirst('=', line)
        eq === nothing && continue
        key = strip(line[1:eq-1]); val = strip(line[eq+1:end])
        isempty(key) || (mph[String(key)] = String(val))
    end
    return mph
end

# ── GIADR scale factors → per-sample radiance scale over the full 8700 grid ───

function _parse_scalefactors(rec::Vector{UInt8})::Vector{Float64}
    nb = Int(_be_i2(rec, 20))
    (1 <= nb <= 10) || error("IASI L1C: implausible scale-band count $nb in GIADR")
    nsfirst = [Int(_be_i2(rec, 22 + 2*(k-1))) for k in 1:10]
    nslast  = [Int(_be_i2(rec, 42 + 2*(k-1))) for k in 1:10]
    expo    = [Int(_be_i2(rec, 62 + 2*(k-1))) for k in 1:10]
    scale = ones(Float64, _IASI_NSPEC)
    for b in 1:nb
        lo = clamp(nsfirst[b], 1, _IASI_NSPEC)
        hi = clamp(nslast[b],  1, _IASI_NSPEC)
        lo <= hi && (scale[lo:hi] .= 10.0^(-expo[b]))
    end
    return scale
end

# Default IASI L1C scale bands (used only if no GIADR scale record is present).
function _default_scale()::Vector{Float64}
    bands = ((1, 1990, 7), (1991, 5116, 8), (5117, 8461, 9))   # (first, last, exp)
    scale = ones(Float64, _IASI_NSPEC)
    for (lo, hi, e) in bands
        scale[lo:hi] .= 10.0^(-e)
    end
    return scale
end

# ── Reader ─────────────────────────────────────────────────────────────────────

"""
    read_iasi_l1c(path; bright=false, wnolim=(645.0, 2760.0),
                  cldlim=nothing, latlim=nothing, lonlim=nothing,
                  zenlim=nothing, szalim=nothing, max_fov=nothing,
                  validate=true) -> IASIL1CGranule

Read an IASI L1C EPS-native `.nat` granule, returning the FOVs passing the
selection filters. Mirrors the options of Dudhia's `read_iasi_l1c.py`.

# Keywords
- `bright`  : return brightness temperature [K] instead of radiance.
- `wnolim`  : `(ν_lo, ν_hi)` channel window (cm⁻¹); spectra are sliced to it.
- `cldlim`  : `(lo, hi)` AVHRR cloud-fraction window (%); e.g. `(0, 20)` keeps
              near-clear FOVs. `nothing` = no cloud screening.
- `latlim`, `lonlim`, `zenlim`, `szalim` : `(lo, hi)` geographic / geometry filters.
- `sralim`  : `(lo, hi)` solar-reflection-angle window (deg). To reject sun glint,
              keep FOVs *away* from the specular path, e.g. `sralim=(15, 180)`
              excludes the 15° glint cone. `nothing` = no glint screening.
- `max_fov` : stop after retaining this many FOVs (quick look / testing).
- `validate`: sanity-check lat/lon ranges and (when radiances are read) the BT
              range; raises if a parsed field is implausible (offset drift guard).

The internal binary layout is ported from the Oxford EODG reader; see the module
docstring. Radiances are in mW/(m²·sr·cm⁻¹) (the `iasi_forward_model` convention).
"""
function read_iasi_l1c(path::AbstractString;
                       bright::Bool = false,
                       wnolim::Tuple{Real,Real} = (_IASI_WNO1, 2760.0),
                       cldlim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       latlim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       lonlim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       zenlim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       szalim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       sralim::Union{Nothing,Tuple{Real,Real}} = nothing,
                       max_fov::Union{Nothing,Integer} = nothing,
                       validate::Bool = true)::IASIL1CGranule
    # Channel window → indices into the 8461-channel grid (645 + 0.25·n).
    i0 = round(Int, (wnolim[1] - _IASI_WNO1) / _IASI_WNOD)
    i1 = round(Int, (wnolim[2] - _IASI_WNO1) / _IASI_WNOD)
    (0 <= i0 <= i1 <= _IASI_NWNO - 1) ||
        error("wnolim $wnolim outside the IASI grid 645–2760 cm⁻¹")
    wno = [_IASI_WNO1 + n * _IASI_WNOD for n in i0:i1]
    nwno = length(wno)

    inwin(x, lim) = lim === nothing || (lim[1] <= x <= lim[2])

    mph    = Dict{String,String}()
    scale  = Float64[]          # per-sample radiance scale (8700), from GIADR
    cols_R = Vector{Float64}[]  # retained spectra
    lat = Float64[]; lon = Float64[]
    zen = Float64[]; azi = Float64[]; sza = Float64[]; saa = Float64[]; sra = Float64[]
    cld = Float64[]; lnd = Float64[]; line = Int[]; step = Int[]; pix = Int[]

    open(path) do io
        seekend(io); fsize = position(io); seekstart(io)
        nline = 0
        while position(io) + _EPS_GRH_SIZE <= fsize
            pos = position(io)
            grh = read(io, _EPS_GRH_SIZE)
            rclass   = Int(grh[1]); subclass = Int(grh[3])
            rsize    = Int(_be_u4(grh, 4))
            rsize < _EPS_GRH_SIZE && break          # corrupt / trailing garbage
            seek(io, pos)
            rec = read(io, rsize)
            length(rec) == rsize || break

            if rclass == _EPS_CLASS_MPHR
                mph = _parse_mphr(rec[_EPS_GRH_SIZE+1:end])
            elseif rclass == _EPS_CLASS_GIADR && (subclass == 1 || rsize == 84)
                isempty(scale) && (scale = _parse_scalefactors(rec))
            elseif rclass == _EPS_CLASS_MDR && subclass == _EPS_MDR_SUBCLASS
                nline += 1
                isempty(scale) && (scale = _default_scale())
                _extract_mdr!(rec, nline, scale, i0, i1, bright, nwno,
                              cldlim, latlim, lonlim, zenlim, szalim, sralim, inwin,
                              cols_R, lat, lon, zen, azi, sza, saa, sra,
                              cld, lnd, line, step, pix)
                max_fov !== nothing && length(lat) >= max_fov && break
            end
        end
    end

    n = length(lat)
    spc = Matrix{Float64}(undef, nwno, n)
    @inbounds for j in 1:n
        spc[:, j] .= cols_R[j]
    end

    if validate && n > 0
        (all(x -> -90.0 <= x <= 90.0, lat) && all(x -> -180.0 <= x <= 180.0, lon)) ||
            error("IASI L1C: lat/lon out of range — MDR offsets likely don't match " *
                  "this product version (geoloc parse). Verify against the granule.")
        if !bright && nwno > 0
            bt = brightness_temperature(wno[1], spc[1, 1])
            (100.0 <= bt <= 400.0) ||
                error("IASI L1C: BT=$(round(bt;digits=1)) K out of 100–400 K — " *
                      "radiance scaling/offsets likely wrong. Verify against the granule.")
        end
    end

    return IASIL1CGranule(String(path), mph, wno, spc, bright,
                          lat, lon, zen, azi, sza, saa, sra, cld, lnd, line, step, pix)
end

# Extract & filter one MDR scan line, pushing retained FOVs onto the accumulators.
function _extract_mdr!(rec, nline, scale, i0, i1, bright, nwno,
                       cldlim, latlim, lonlim, zenlim, szalim, sralim, inwin,
                       cols_R, lat, lon, zen, azi, sza, saa, sra, cld, lnd, line, step, pix)
    @inbounds for s in 0:_IASI_NSTP-1, p in 0:_IASI_NPIX-1
        q = s * _IASI_NPIX + p
        lonv = _be_i4(rec, _OFF_GEOLOC  + (q*2    )*4) * 1.0e-6
        latv = _be_i4(rec, _OFF_GEOLOC  + (q*2 + 1)*4) * 1.0e-6
        zenv = _be_i4(rec, _OFF_ANG_SAT + (q*2    )*4) * 1.0e-6   # θᵢ
        aziv = _be_i4(rec, _OFF_ANG_SAT + (q*2 + 1)*4) * 1.0e-6   # ϕᵢ
        szav = _be_i4(rec, _OFF_ANG_SUN + (q*2    )*4) * 1.0e-6   # θₛ
        saav = _be_i4(rec, _OFF_ANG_SUN + (q*2 + 1)*4) * 1.0e-6   # ϕₛ
        srav = _solar_reflection_angle(zenv, aziv, szav, saav)   # φᵣ (sun glint)
        cldv = Float64(_u1(rec, _OFF_CLDFRAC + q))
        lndv = Float64(_u1(rec, _OFF_LNDFRAC + q))

        (inwin(latv, latlim) && inwin(lonv, lonlim) && inwin(zenv, zenlim) &&
         inwin(szav, szalim) && inwin(srav, sralim) && inwin(cldv, cldlim)) || continue

        # De-scale the requested channel window → mW/(m²·sr·cm⁻¹) (or BT).
        base = _OFF_SPEC + q * _IASI_NSPEC * 2
        R = Vector{Float64}(undef, nwno)
        for (k, n) in enumerate(i0:i1)
            raw = _be_i2(rec, base + n * 2)
            rad = Float64(raw) * scale[n + 1] * _IASI_RADFAC
            R[k] = bright ? brightness_temperature(_IASI_WNO1 + n * _IASI_WNOD, rad) : rad
        end

        push!(cols_R, R)
        push!(lat, latv); push!(lon, lonv)
        push!(zen, zenv); push!(azi, aziv); push!(sza, szav); push!(saa, saav); push!(sra, srav)
        push!(cld, cldv); push!(lnd, lndv)
        push!(line, nline); push!(step, s + 1); push!(pix, p + 1)
    end
end

"""
    _solar_reflection_angle(θᵢ, ϕᵢ, θₛ, ϕₛ) -> Float64

Solar reflection (sun-glint) angle φᵣ in degrees — the angle between the IASI
view direction Î and the specular solar-reflection direction R̂ off a flat
surface (Vincent thesis Eq. 2.4):

    cos φᵣ = R̂·Î = cos θₛ cos θᵢ − sin θₛ sin θᵢ cos(ϕₛ − ϕᵢ).

All inputs in degrees. φᵣ → 0 means IASI looks straight down the specular path
(maximum sun glint); reject the glint cone with the reader's `sralim` keyword.
"""
@inline function _solar_reflection_angle(θI::Float64, ϕI::Float64,
                                         θs::Float64, ϕs::Float64)::Float64
    c = cosd(θs) * cosd(θI) - sind(θs) * sind(θI) * cosd(ϕs - ϕI)
    return acosd(clamp(c, -1.0, 1.0))
end

# ── Convenience accessors ────────────────────────────────────────────────────

"""
    measurement(granule, i) -> (ν::Vector{Float64}, R::Vector{Float64})

The wavenumber grid and measured spectrum of FOV `i`, ready to feed
`optimal_estimation` as the observation `y` (radiance, or BT if the granule was
read with `bright=true`).
"""
measurement(g::IASIL1CGranule, i::Integer) = (g.wno, g.spc[:, i])

"""
    cloud_fraction(granule) -> Vector{Float64}

The per-FOV AVHRR cloud fraction (%) — use it to select clear scenes, e.g.
`read_iasi_l1c(path; cldlim=(0, 10))` or post-hoc on `granule.cld`.
"""
cloud_fraction(g::IASIL1CGranule) = g.cld

"""
    solar_reflection_angle(granule) -> Vector{Float64}

The per-FOV solar reflection (sun-glint) angle φᵣ (deg) — the angle between the
IASI view and the specular solar-reflection path (Vincent thesis Eq. 2.4). Small
values indicate sun glint; screen with `read_iasi_l1c(path; sralim=(15, 180))` or
post-hoc on `granule.sra`.
"""
solar_reflection_angle(g::IASIL1CGranule) = g.sra

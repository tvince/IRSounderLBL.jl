# Synthetic EPS-native round-trip: build a tiny .nat (MPHR + GIADR scale +
# one MDR) with planted values at Dudhia's offsets, read it back, verify.
using IRSounderLBL
const M = IRSounderLBL

put_i4!(b, o, v) = (b[o+1:o+4] = reinterpret(UInt8, [hton(Int32(v))]))
put_i2!(b, o, v) = (b[o+1:o+2] = reinterpret(UInt8, [hton(Int16(v))]))
put_u4!(b, o, v) = (b[o+1:o+4] = reinterpret(UInt8, [hton(UInt32(v))]))

function grh(class, subclass, rsize)
    g = zeros(UInt8, 20)
    g[1] = UInt8(class); g[2] = UInt8(class); g[3] = UInt8(subclass); g[4] = 0x05
    put_u4!(g, 4, rsize)
    return g
end

# MPHR record
mphr_payload = Vector{UInt8}(codeunits("INSTRUMENT_ID = IASI\nSPACECRAFT_ID = M02\n"))
mphr = vcat(grh(1, 0, 20 + length(mphr_payload)), mphr_payload)

# GIADR scale-factor record (84 bytes): one band, channels 1..8700, exp=7
giadr = vcat(grh(5, 1, 84), zeros(UInt8, 64))
put_i2!(giadr, 20, 1)        # NbScale = 1
put_i2!(giadr, 22, 1)        # Nsfirst[1] = 1
put_i2!(giadr, 42, 8700)     # Nslast[1]  = 8700
put_i2!(giadr, 62, 7)        # ScaleFactor[1] = 7  → 10^-7

# MDR record (full size), with planted FOVs
MDR_SIZE = 2728908
mdr = zeros(UInt8, MDR_SIZE)
mdr[1:20] = grh(8, 2, MDR_SIZE)
# IDefNsfirst1b / Nslast1b (valid 8461 block = samples 1..8461)
put_i4!(mdr, M._OFF_NSFIRST, 1); put_i4!(mdr, M._OFF_NSLAST, 8461)

planted = Dict{Tuple{Int,Int},NamedTuple}()
for s in 0:29, p in 0:3
    q = s*4 + p
    lon = -120.0 - q*0.001; lat = 30.0 + q*0.002
    zen = 5.0 + p; sza = 40.0 + s*0.1
    cldpct = (q % 101)               # 0..100
    put_i4!(mdr, M._OFF_GEOLOC  + (q*2  )*4, round(Int, lon/1e-6))
    put_i4!(mdr, M._OFF_GEOLOC  + (q*2+1)*4, round(Int, lat/1e-6))
    azi = 100.0 + q*0.05; saa = 160.0 - q*0.1
    put_i4!(mdr, M._OFF_ANG_SAT + (q*2  )*4, round(Int, zen/1e-6))
    put_i4!(mdr, M._OFF_ANG_SAT + (q*2+1)*4, round(Int, azi/1e-6))
    put_i4!(mdr, M._OFF_ANG_SUN + (q*2  )*4, round(Int, sza/1e-6))
    put_i4!(mdr, M._OFF_ANG_SUN + (q*2+1)*4, round(Int, saa/1e-6))
    mdr[M._OFF_CLDFRAC + q + 1] = UInt8(cldpct)
    mdr[M._OFF_LNDFRAC + q + 1] = UInt8(50)
    # spectrum: raw=7000 → rad = 7000 × 1e-7 × 1e5 = 70.0 mW/(m²·sr·cm⁻¹)
    base = M._OFF_SPEC + q*8700*2
    for n in 0:8460
        put_i2!(mdr, base + n*2, 7000)
    end
    planted[(s,p)] = (; lon, lat, zen, sza, azi, saa, cld=Float64(cldpct))
end

path = tempname() * ".nat"
open(path, "w") do io
    write(io, mphr); write(io, giadr); write(io, mdr)
end

println("== full read ==")
g = read_iasi_l1c(path)
println(g)
@assert nfov(g) == 30*4
@assert g.mph["INSTRUMENT_ID"] == "IASI"
@assert length(g.wno) == 8461 && g.wno[1] == 645.0 && g.wno[end] == 2760.0
@assert all(≈(70.0), g.spc)                       # de-scaling exact
# geoloc round-trip for a sample FOV
i = findfirst(k -> g.step[k]==1 && g.pix[k]==1, 1:nfov(g))
pl = planted[(0,0)]
@assert isapprox(g.lat[i], pl.lat; atol=1e-6) && isapprox(g.lon[i], pl.lon; atol=1e-6)
@assert isapprox(g.zen[i], pl.zen; atol=1e-6) && isapprox(g.sza[i], pl.sza; atol=1e-6)
# solar reflection angle φr (Eq. 2.4)
φexp = acosd(cosd(g.sza[i])*cosd(g.zen[i]) - sind(g.sza[i])*sind(g.zen[i])*cosd(g.saa[i]-g.azi[i]))
@assert isapprox(g.sra[i], φexp; atol=1e-6)
@assert isapprox(M._solar_reflection_angle(30.0,200.0,30.0,20.0), 0.0; atol=1e-3)   # specular
println("  lat/lon/zen/sza/azi/saa round-trip OK; spc=70.0 exact; sra[$i]=$(round(g.sra[i];digits=3))° (φr range $(round(minimum(g.sra);digits=2))–$(round(maximum(g.sra);digits=2))°)")

println("== cloud screen (cldlim=(0,20)) ==")
gc = read_iasi_l1c(path; cldlim=(0, 20))
@assert all(c -> 0 <= c <= 20, gc.cld)
@assert nfov(gc) < nfov(g)
println("  kept $(nfov(gc))/$(nfov(g)) FOVs; max cld=$(maximum(gc.cld))%")

println("== sun-glint screen (sralim) ==")
lo, hi = extrema(g.sra); mid = (lo+hi)/2
gg = read_iasi_l1c(path; sralim=(0, mid))
@assert 0 < nfov(gg) < nfov(g) && all(a -> a <= mid, gg.sra)
println("  kept $(nfov(gg))/$(nfov(g)) FOVs with φr ≤ $(round(mid;digits=2))°")

println("== wnolim window + bright ==")
gw = read_iasi_l1c(path; wnolim=(700.0, 710.0), bright=true)
@assert gw.wno[1] == 700.0 && gw.wno[end] == 710.0
@assert gw.is_bt && all(bt -> 100 < bt < 400, gw.spc)
println("  window $(length(gw.wno)) ch, BT range $(round(minimum(gw.spc);digits=2))–$(round(maximum(gw.spc);digits=2)) K")

println("\nALL SYNTHETIC CHECKS PASSED")
rm(path)

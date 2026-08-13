# Compute scene-specific NEΔT (thesis Eq. 2.3) across IASI band 1 for a real FOV,
# to visualise the cold-CO2-band inflation vs the flat NEΔT_280K reference.
# Dumps data/scene_nedt.csv → plot with plot_scene_nedt.py.

using IRSounderLBL
using Printf

const GRANULE = "data/iasi_l1c/IASI_xxx_1C_M03_20260609152958Z_20260609153253Z_N_O_20260609165144Z__20260609165227"
const ν_LO, ν_HI = 645.0, 1210.0     # IASI band 1
const NEDT_280 = 0.25

g = read_iasi_l1c(GRANULE; bright=true, wnolim=(ν_LO, ν_HI),
                  cldlim=(0.0, 5.0), sralim=(15.0, 180.0), zenlim=(0.0, 25.0), max_fov=400)
ifov = argmin(cloud_fraction(g))
νobs, bt = measurement(g, ifov)
σ_scene = scene_nedt(νobs, bt; nedt_280K=NEDT_280, T_ref=280.0)
@printf("FOV #%d  %d channels  BT %.1f…%.1f K;  scene σ_BT %.2f…%.2f K\n",
        ifov, length(νobs), minimum(bt), maximum(bt), minimum(σ_scene), maximum(σ_scene))

open("data/scene_nedt.csv", "w") do io
    println(io, "wavenumber_cm-1,bt_obs_K,nedt_scene_K,nedt_ref_K")
    for i in eachindex(νobs)
        @printf(io, "%.4f,%.4f,%.4f,%.4f\n", νobs[i], bt[i], σ_scene[i], NEDT_280)
    end
end
println("wrote data/scene_nedt.csv")

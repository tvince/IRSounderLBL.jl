using IRSounderLBL

# ─── Setup ───────────────────────────────────────────────────────────────────
ll   = load_hitran_par("data/co2_645_700.par"; ν_min=645.0, ν_max=700.0)
prof = us_standard_atmosphere()
layers = layer_properties(prof)

println("=== Atmosphere temperature profile (first 10 layers) ===")
for k in 1:10
    println("  layer $k: p=$(round(layers.p_mid[k];digits=1)) hPa, T=$(round(layers.T_mid[k];digits=1)) K, Δp=$(round(layers.Δp[k];digits=2)) hPa")
end
println("  ...")
for k in 40:42
    println("  layer $k: p=$(round(layers.p_mid[k];digits=4)) hPa, T=$(round(layers.T_mid[k];digits=1)) K")
end

# ─── Cross sections and column density for surface layer ─────────────────────
ν_test = wavenumber_grid(660.0, 670.0; n=11)
k = 1   # surface layer
vmr_co2 = layers.vmr_mid[CO2][k]
p_atm_k = layers.p_mid[k] / 1013.25
T_k      = layers.T_mid[k]
Δp_k     = layers.Δp[k]

σ = compute_voigt_cross_sections(ν_test, ll, T_k, p_atm_k)
Nair = 2.1521e21
N_col = vmr_co2 * Δp_k * Nair

println("\n=== Surface layer (k=1) cross-sections 660–670 cm⁻¹ ===")
println("  vmr_co2=$(vmr_co2), p=$(round(layers.p_mid[k];digits=1)) hPa, T=$(T_k) K, Δp=$(Δp_k) hPa")
println("  N_col (code) = $N_col  molec/cm²")
Nair_correct = 2.1209e22
println("  N_col (correct) = $(vmr_co2 * Δp_k * Nair_correct)  molec/cm²")
println("  max σ = $(maximum(σ)) cm²/molec")
println("  max τ (code)    = $(maximum(σ) * N_col)")
println("  max τ (correct) = $(maximum(σ) * vmr_co2 * Δp_k * Nair_correct)")

# ─── Check total column τ at band center ─────────────────────────────────────
ν_center = wavenumber_grid(666.5, 667.5; n=5)
τ_col = zeros(5)
for k in 1:length(layers.p_mid)
    vmr_k   = layers.vmr_mid[CO2][k]
    p_atm_k = layers.p_mid[k] / 1013.25
    T_k     = layers.T_mid[k]
    Δp_k    = layers.Δp[k]
    σ_k = compute_voigt_cross_sections(ν_center, ll, T_k, p_atm_k)
    N_col_k = vmr_k * Δp_k * Nair
    τ_col .+= σ_k .* N_col_k
end
println("\n=== Total column optical depth at CO₂ band center (667 cm⁻¹) ===")
println("  τ = $(τ_col)")

# ─── BT in transparent window ────────────────────────────────────────────────
linelists = Dict{GasSpecies, HITRANLinelist}(CO2 => ll)
ν_iasi, R_iasi, BT_iasi = iasi_forward_model(prof, linelists)

mask_window = (ν_iasi.ν .>= 800.0) .& (ν_iasi.ν .<= 1000.0)
BT_window = BT_iasi[mask_window]
println("\n=== BT in 800–1000 cm⁻¹ transparent window ===")
println("  mean BT = $(round(sum(BT_window)/length(BT_window); digits=2)) K  (expect ~288 K)")
println("  min  BT = $(round(minimum(BT_window); digits=2)) K")
println("  max  BT = $(round(maximum(BT_window); digits=2)) K")

# ─── Planck sanity check ──────────────────────────────────────────────────────
println("\n=== Planck function sanity check ===")
B_sfc  = planck_radiance(667.0, 288.15)
B_cold = planck_radiance(667.0, 165.0)
println("  B(667 cm⁻¹, 288 K) = $B_sfc  mW/m²/sr/cm⁻¹")
println("  B(667 cm⁻¹, 165 K) = $B_cold  mW/m²/sr/cm⁻¹")

idx_667 = argmin(abs.(ν_iasi.ν .- 667.0))
println("  Actual R at ~667 cm⁻¹ = $(R_iasi[idx_667])")
println("  Actual BT at ~667 cm⁻¹ = $(BT_iasi[idx_667]) K")

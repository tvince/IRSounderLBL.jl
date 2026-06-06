"""
Plot Julia vs LBLRTM Curtis-Godson per-layer values for the AFGL US Standard
50-level / 4.3 µm comparison.

Five panels, all with a Julia-minus-LBLRTM diff overlay:
  1.  T_eff(z)           — effective layer temperature
  2.  log10 p_eff(z)     — effective layer pressure
  3.  log10 N_CO2(z)     — CO₂ column density per layer
  4.  γ_L(z) at 2364 cm⁻¹ — Lorentz HWHM at layer (p_eff, T_eff)
  5.  γ_D(z) at 2364 cm⁻¹ — Doppler HWHM at layer T_eff

For (4) and (5), LBLRTM's printed ALPHL/ALPHD are for a reference line with
ALFAL0=0.04 cm⁻¹/atm at standard conditions (not our 2364 cm⁻¹ line). We
RESCALE LBLRTM's ALPHL to the 2364 cm⁻¹ line's γ_air = 0.0722 and n_air =
0.710 read from HITRAN:
    γ_L_2364 = ALPHL_ref × (γ_air_2364 / 0.04) × (T_ref/T)^(n_air_2364 - 0.5)
LBLRTM's reference ALPHD is for an unspecified molecular mass, so we rescale
to CO₂ (M=44.010 amu) by the Doppler-width prefactor ratio at line ν₀=2364.1.

If T_eff, p_eff, and N_CO2 agree closely between codes, the CG layer
averaging is NOT the source of the per-line core-temperature comb at line
centers in the 2355-2375 cm⁻¹ window.

Usage:
  python scripts/compare_lblrtm_julia_layers.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LBL = np.genfromtxt("data/lblrtm/lblrtm_layers_43um.csv",
                    delimiter=",", names=True)
JUL = np.genfromtxt("data/lblrtm/julia_layers_43um.csv",
                    delimiter=",", names=True)

assert len(LBL) == len(JUL), f"row mismatch: LBLRTM={len(LBL)} Julia={len(JUL)}"
n = len(LBL)
z_mid = 0.5 * (LBL["z_from_km"] + LBL["z_to_km"])

# Sanity: layer indices should align 1..49 in z order
assert np.allclose(z_mid, 0.5 * (JUL["z_from_km"] + JUL["z_to_km"]))

# ── Rescale LBLRTM's ref ALPHL to the 2364 cm⁻¹ line --------------------
# ALPHL_ref = ALFAL0 × (P/P0) × (T0/T)^n_ref  with ALFAL0=0.04, n_ref=0.5
# (LBLRTM's ALFAL0 convention; n is the molecular average in DEFAULT mode)
ALFAL0   = 0.04
N_REF    = 0.5
GAMMA_AIR_2364 = 0.0722
N_AIR_2364     = 0.710
T_REF          = 296.0

# γ_L_2364(P,T) = γ_air × (P/P0) × (T_ref/T)^n_air
# We can compute it directly from the LBLRTM (p_eff, T_eff) — no need to
# touch ALPHL.
alpha_L_lblrtm_2364 = (GAMMA_AIR_2364 * (LBL["p_eff_hPa"] / 1013.25)
                       * (T_REF / LBL["T_eff_K"]) ** N_AIR_2364)

# Doppler width for the same line (γ_D = ν₀ × 3.581e-7 × sqrt(T/M))
NU0_2364 = 2364.1053
M_CO2    = 44.010
alpha_D_lblrtm_2364 = (NU0_2364 * 3.58126e-7
                       * np.sqrt(LBL["T_eff_K"] / M_CO2))

# Julia values are already at the 2364 cm⁻¹ line from the dump script
alpha_L_julia = JUL["gamma_L_2364"]
alpha_D_julia = JUL["gamma_D_2364"]


def panel(ax, ax_d, y_l, y_j, ylabel, title, log=False):
    ax.plot(z_mid, y_l, color="k",        lw=1.2, marker="o", ms=3, label="LBLRTM")
    ax.plot(z_mid, y_j, color="tab:red",  lw=1.2, marker="o", ms=3,
            alpha=0.7, label="Julia (CG)")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    d = y_j - y_l
    rms = np.sqrt(np.mean(d ** 2))
    ax_d.plot(z_mid, d, color="tab:blue", lw=1.0, marker="o", ms=2)
    ax_d.axhline(0, color="grey", lw=0.5)
    ax_d.set_ylabel(f"Julia − LBLRTM\nRMS {rms:.3g}", fontsize=8)
    ax_d.grid(True, alpha=0.3)
    ax_d.axvspan(50, 95, color="orange", alpha=0.08)   # mesopause band


fig = plt.figure(figsize=(14, 13))
gs  = fig.add_gridspec(10, 1, hspace=0.6)

panels_spec = [
    (LBL["T_eff_K"],      JUL["T_eff_K"],     "T$_{eff}$ [K]",
     "Curtis–Godson effective temperature per layer", False),
    (LBL["p_eff_hPa"],    JUL["p_eff_hPa"],   "p$_{eff}$ [hPa]",
     "Curtis–Godson effective pressure per layer",     True),
    (LBL["N_CO2_cm2"],    JUL["N_CO2_cm2"],   "N$_{CO_2}$ [molec cm$^{-2}$]",
     "CO$_2$ column density per layer",                True),
    (alpha_L_lblrtm_2364, alpha_L_julia,
     "γ$_L$ [cm$^{-1}$]",
     "Lorentz HWHM of 2364.1 cm$^{-1}$ CO$_2$ line at layer (P,T)", True),
    (alpha_D_lblrtm_2364, alpha_D_julia,
     "γ$_D$ [cm$^{-1}$]",
     "Doppler HWHM of 2364.1 cm$^{-1}$ CO$_2$ line at layer T",     True),
]

axes_main, axes_diff = [], []
for i in range(5):
    am = fig.add_subplot(gs[2*i,     0])
    ad = fig.add_subplot(gs[2*i + 1, 0], sharex=am)
    axes_main.append(am); axes_diff.append(ad)

for am, ad, (y_l, y_j, ylab, title, log) in zip(axes_main, axes_diff, panels_spec):
    panel(am, ad, y_l, y_j, ylab, title, log=log)
    am.tick_params(labelbottom=False)

axes_diff[-1].set_xlabel("layer mid-altitude [km]")

fig.suptitle("Julia (CG) vs LBLRTM (TAPE6) per-layer effective values, "
             "4.3 µm comparison\n"
             "(shaded band: 50–95 km, where strong 4.3 µm CO$_2$ line cores saturate)",
             fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "data/lblrtm/julia_lblrtm_layer_diff.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")

# Print a quick numeric summary for the 50–95 km band
m = (z_mid >= 50) & (z_mid <= 95)
print("\n=== Diff Julia − LBLRTM, 50–95 km band ===")
for label, l, j in [
    ("T_eff [K]",        LBL["T_eff_K"],     JUL["T_eff_K"]),
    ("p_eff [hPa]",      LBL["p_eff_hPa"],   JUL["p_eff_hPa"]),
    ("N_CO2 [cm⁻²]",     LBL["N_CO2_cm2"],   JUL["N_CO2_cm2"]),
    ("γ_L_2364 [cm⁻¹]",  alpha_L_lblrtm_2364, alpha_L_julia),
    ("γ_D_2364 [cm⁻¹]",  alpha_D_lblrtm_2364, alpha_D_julia),
]:
    d = (j - l)[m]
    rel = d / l[m]
    print(f"  {label:20s}  mean Δ={d.mean():+.3e}  "
          f"max|Δ|={np.abs(d).max():.3e}  "
          f"mean rel={rel.mean()*100:+.2f}%")

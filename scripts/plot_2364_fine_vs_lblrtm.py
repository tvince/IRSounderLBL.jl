"""
Phase-2 plot: does the +7 K Julia−LBLRTM spike at 2364 cm⁻¹ collapse when
Julia's wavenumber grid is refined from 0.005 to 0.0005 cm⁻¹?

Compares three spectra in the 2355–2375 cm⁻¹ window (cont-OFF, pure Voigt):
  - LBLRTM       (DV ≈ 0.0005 cm⁻¹, panel scheme)
  - Julia coarse (DV = 0.005 cm⁻¹)   — current baseline
  - Julia fine   (DV = 0.0005 cm⁻¹)  — phase-2 refined grid

Top panel:    absolute BT, all three.
Middle panel: Julia coarse − LBLRTM (the +7 K spike at 2364 cm⁻¹).
Bottom panel: Julia fine   − LBLRTM (does it shrink?).

Usage:
  python scripts/plot_2364_fine_vs_lblrtm.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"
W_MIN, W_MAX = 2355.0, 2375.0


def load(path, nu_col, bt_col):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu_col], a[bt_col]


def window(nu, y, lo, hi):
    m = (nu >= lo) & (nu <= hi)
    return nu[m], y[m]


def main():
    nl, bt_l = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv",
                    "wavenumber_cm1", "BT_K")
    nj_c, bt_jc = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")
    nj_f, bt_jf = load(f"{DATA}/julia_bt_2364_fine_contOFF.csv", "nu_cm1", "BT_K")

    nl,   bt_l   = window(nl,   bt_l,   W_MIN, W_MAX)
    nj_c, bt_jc  = window(nj_c, bt_jc,  W_MIN, W_MAX)
    nj_f, bt_jf  = window(nj_f, bt_jf,  W_MIN, W_MAX)

    # Interpolate LBLRTM onto each Julia grid for differencing
    bt_l_on_c = np.interp(nj_c, nl, bt_l)
    bt_l_on_f = np.interp(nj_f, nl, bt_l)

    d_c = bt_jc - bt_l_on_c
    d_f = bt_jf - bt_l_on_f

    def stats(label, d):
        rms = np.sqrt((d**2).mean())
        i = np.argmax(np.abs(d))
        print(f"  {label:25s} RMS {rms:6.3f} K  worst {d[i]:+6.2f} K")
        return rms, i

    print("=== 2355–2375 cm⁻¹, cont-OFF (Julia − LBLRTM) ===")
    rms_c, ic = stats("Julia Δν=0.005 (coarse)", d_c)
    rms_f, ifi = stats("Julia Δν=0.0005 (fine)", d_f)
    print(f"\n  ratio fine/coarse RMS = {rms_f/rms_c:.3f}")
    print(f"  spike peak coarse: {d_c[ic]:+.2f} K @ {nj_c[ic]:.3f} cm⁻¹")
    print(f"  spike peak fine  : {d_f[ifi]:+.2f} K @ {nj_f[ifi]:.3f} cm⁻¹")

    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    ax[0].plot(nl,   bt_l,  lw=0.5, color="k",        label="LBLRTM ($\\Delta\\nu$≈0.0005)")
    ax[0].plot(nj_c, bt_jc, lw=0.5, color="tab:red",  alpha=0.85,
               label="Julia coarse ($\\Delta\\nu$=0.005)")
    ax[0].plot(nj_f, bt_jf, lw=0.5, color="tab:blue", alpha=0.85,
               label="Julia fine   ($\\Delta\\nu$=0.0005)")
    ax[0].set_ylabel("BT [K]")
    ax[0].set_title("CO$_2$ ν₃ band head, 4.3 µm, cont-OFF (pure Voigt)")
    ax[0].legend(loc="best", fontsize=9)

    ax[1].plot(nj_c, d_c, lw=0.4, color="tab:red")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_ylabel("ΔBT [K]")
    ax[1].set_title(f"Julia coarse − LBLRTM    RMS = {rms_c:.3f} K   "
                    f"worst {d_c[ic]:+.2f} K @ {nj_c[ic]:.2f} cm⁻¹")

    ax[2].plot(nj_f, d_f, lw=0.4, color="tab:blue")
    ax[2].axhline(0, color="grey", lw=0.5)
    ax[2].set_xlabel("wavenumber [cm⁻¹]")
    ax[2].set_ylabel("ΔBT [K]")
    ax[2].set_title(f"Julia fine − LBLRTM    RMS = {rms_f:.3f} K   "
                    f"worst {d_f[ifi]:+.2f} K @ {nj_f[ifi]:.2f} cm⁻¹")

    # Same y-range for the two diff panels to make the shrinkage visible
    ymax = max(abs(d_c).max(), abs(d_f).max()) * 1.1
    ax[1].set_ylim(-ymax, ymax)
    ax[2].set_ylim(-ymax, ymax)

    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_2364_fine_vs_coarse.png"
    plt.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

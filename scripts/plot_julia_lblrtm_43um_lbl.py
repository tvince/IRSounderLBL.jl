"""
LBL-only (cont-OFF) Julia vs LBLRTM over the 4.3 µm CO₂ band (2000-2500 cm⁻¹).

Top panel:    absolute brightness temperatures, both codes.
Bottom panel: difference (Julia − LBLRTM).

Unlike compare_julia_lblrtm_43um.py (which differences the continuum effect
within each code, cancelling common LBL features), this shows the raw pure-Voigt
LBL agreement — so the +7 K Julia−LBLRTM spike at the 2364 cm⁻¹ ν₃ Q-branch /
band-head is visible directly.

Usage:
  python scripts/plot_julia_lblrtm_43um_lbl.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"


def load(path, nu_col, bt_col):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu_col], a[bt_col]


def main():
    nj, j_off = load(f"{DATA}/julia_bt_43um_contOFF.csv", "nu_cm1", "BT_K")
    nl, l_off = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv", "wavenumber_cm1", "BT_K")
    L_off = np.interp(nj, nl, l_off)

    d = j_off - L_off
    rms = np.sqrt((d**2).mean())
    imax = np.argmax(np.abs(d))
    print("=== LBL-only (cont-OFF) BT agreement, Julia − LBLRTM ===")
    print(f"  RMS {rms:.3f} K   min {d.min():+.2f}  max {d.max():+.2f} K")
    print(f"  worst |Δ| = {d[imax]:+.2f} K at {nj[imax]:.2f} cm⁻¹")

    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(nj, L_off, lw=0.4, color="k", label="LBLRTM")
    ax[0].plot(nj, j_off, lw=0.4, color="tab:red", alpha=0.8, label="Julia")
    ax[0].set_ylabel("BT [K]")
    ax[0].set_title("LBL-only (cont-OFF) CO₂ brightness temperature, 4.3 µm")
    ax[0].legend(loc="best")

    ax[1].plot(nj, d, lw=0.3, color="tab:blue")
    ax[1].axhline(0, color="grey", lw=0.5)
    ax[1].set_title(f"difference (Julia − LBLRTM), RMS={rms:.3f} K")
    ax[1].set_xlabel("wavenumber [cm⁻¹]"); ax[1].set_ylabel("ΔBT [K]")
    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_lbl.png"
    plt.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

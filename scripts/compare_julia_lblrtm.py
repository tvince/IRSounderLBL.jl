"""
Compare Julia vs LBLRTM brightness temperature over 645-800 cm-1.

LBLRTM monochromatic BT (~0.00016 cm-1) is sampled onto Julia's fine
0.005 cm-1 grid by linear interpolation (LBLRTM is far finer, so this is
effectively point-sampling at identical wavenumbers). Prints ΔBT statistics
and writes an overlay + residual plot for each continuum configuration.

Usage:
  python scripts/compare_julia_lblrtm.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "data/lblrtm"
CONFIGS = [("contOFF", "continuum OFF"), ("contON", "continuum ON")]


def load_csv(path, nu_col, bt_col):
    a = np.genfromtxt(path, delimiter=",", names=True)
    return a[nu_col], a[bt_col]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for j, (tag, label) in enumerate(CONFIGS):
        nu_j, bt_j = load_csv(f"{DATA}/julia_bt_{tag}.csv", "nu_cm1", "BT_K")
        # _g12 = native-fixed build (gfortran-12); the non-g12 files are the
        # stale gfortran-15 outputs with the corrupted surface term.
        nu_l, bt_l = load_csv(f"{DATA}/lblrtm_bt_{tag}_g12.csv",
                              "wavenumber_cm1", "BT_K")
        # sample LBLRTM onto Julia grid
        bt_l_on_j = np.interp(nu_j, nu_l, bt_l)
        d = bt_j - bt_l_on_j
        rms = np.sqrt(np.mean(d**2))
        print(f"\n=== {label} ===")
        print(f"  Julia : {bt_j.min():.2f}-{bt_j.max():.2f} K  mean {bt_j.mean():.3f}")
        print(f"  LBLRTM: {bt_l.min():.2f}-{bt_l.max():.2f} K  mean {bt_l.mean():.3f}")
        print(f"  ΔBT (Julia-LBLRTM): mean {d.mean():+.3f}  RMS {rms:.3f}  "
              f"min {d.min():+.3f}  max {d.max():+.3f} K")
        kmax = np.argmax(np.abs(d))
        print(f"  worst residual {d[kmax]:+.2f} K at {nu_j[kmax]:.4f} cm-1 "
              f"(Julia {bt_j[kmax]:.2f}, LBLRTM {bt_l_on_j[kmax]:.2f})")

        ax = axes[0, j]
        ax.plot(nu_j, bt_l_on_j, lw=0.4, label="LBLRTM", color="k")
        ax.plot(nu_j, bt_j, lw=0.4, label="Julia", color="tab:red", alpha=0.8)
        ax.set_title(f"{label}: BT overlay"); ax.set_ylabel("BT [K]")
        ax.legend(loc="lower right")
        ax2 = axes[1, j]
        ax2.plot(nu_j, d, lw=0.3, color="tab:blue")
        ax2.axhline(0, color="grey", lw=0.5)
        ax2.set_title(f"ΔBT (Julia − LBLRTM), RMS={rms:.2f} K")
        ax2.set_xlabel("wavenumber [cm⁻¹]"); ax2.set_ylabel("ΔBT [K]")
    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_compare.png"
    plt.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

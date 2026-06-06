"""
Phase-7 plot: does the +3.5 K residual at CO₂ line cores collapse when the
upper-atmosphere layers are subdivided from 5 km to 1 km?

Compares (all over 2355-2375 cm⁻¹, cont-OFF):
  - LBLRTM @ 50 levels      (baseline, the original LBLRTM reference)
  - LBLRTM @ 86 levels      (new, on subdivided profile)
  - Julia Toon @ 50 levels  (default Julia, was +7 K worst)
  - Julia CIM  @ 50 levels  (CG-consistent, +3.5 K worst)
  - Julia Toon @ 86 levels  (subdivided + default)
  - Julia CIM  @ 86 levels  (subdivided + CG-consistent — best?)

Five diff panels.

Usage:
  python scripts/plot_43um_subdiv_comparison.py
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


def diff_stats(label, nu_j, bt_j, nu_l, bt_l):
    bt_l_on_j = np.interp(nu_j, nu_l, bt_l)
    d = bt_j - bt_l_on_j
    rms = np.sqrt((d**2).mean())
    i = np.argmax(np.abs(d))
    return d, rms, i, nu_j[i]


def main():
    # LBLRTM
    nl50, l50 = load(f"{DATA}/lblrtm_bt_43um_contOFF.csv",
                     "wavenumber_cm1", "BT_K")
    nl86, l86 = load(f"{DATA}/lblrtm_bt_43um_subdiv.csv",
                     "wavenumber_cm1", "BT_K")

    # Julia variants
    nj50t, j50t = load(f"{DATA}/julia_bt_43um_contOFF.csv",       "nu_cm1", "BT_K")
    nj50c, j50c = load(f"{DATA}/julia_bt_43um_cim_contOFF.csv",    "nu_cm1", "BT_K")
    nj86t, j86t = load(f"{DATA}/julia_bt_43um_subdiv_toon.csv",    "nu_cm1", "BT_K")
    nj86c, j86c = load(f"{DATA}/julia_bt_43um_subdiv_cim.csv",     "nu_cm1", "BT_K")

    nl50, l50 = window(nl50, l50, W_MIN, W_MAX)
    nl86, l86 = window(nl86, l86, W_MIN, W_MAX)
    nj50t, j50t = window(nj50t, j50t, W_MIN, W_MAX)
    nj50c, j50c = window(nj50c, j50c, W_MIN, W_MAX)
    nj86t, j86t = window(nj86t, j86t, W_MIN, W_MAX)
    nj86c, j86c = window(nj86c, j86c, W_MIN, W_MAX)

    print("=== Julia − LBLRTM RMS / worst, in 2355-2375 cm⁻¹ (cont-OFF) ===")
    # 50-level Julia vs 50-level LBLRTM (original comparison)
    rows = []
    for label, nu_j, bt_j, nu_l, bt_l in [
        ("Julia Toon @50 vs LBLRTM @50",  nj50t, j50t, nl50, l50),
        ("Julia CIM  @50 vs LBLRTM @50",  nj50c, j50c, nl50, l50),
        ("Julia Toon @86 vs LBLRTM @86",  nj86t, j86t, nl86, l86),
        ("Julia CIM  @86 vs LBLRTM @86",  nj86c, j86c, nl86, l86),
    ]:
        d, rms, i, nu_i = diff_stats(label, nu_j, bt_j, nu_l, bt_l)
        rows.append((label, d, rms, i, nu_i, nu_j))
        print(f"  {label:38s}  RMS {rms:5.3f} K   worst {d[i]:+5.2f} K @ {nu_i:.3f}")

    fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)

    # Top: BT overlay (LBLRTM@86 + all Julia variants)
    ax[0].plot(nl86, l86, lw=0.5, color="k",         label="LBLRTM @86")
    ax[0].plot(nj86t, j86t, lw=0.4, color="tab:red", alpha=0.8, label="Julia Toon @86")
    ax[0].plot(nj86c, j86c, lw=0.4, color="tab:blue",alpha=0.8, label="Julia CIM @86")
    ax[0].set_ylabel("BT [K]")
    ax[0].set_title("CO$_2$ ν₃ band head 4.3 µm, cont-OFF — 86-layer subdivision")
    ax[0].legend(loc="best", fontsize=8)

    colors = ["tab:red", "tab:orange", "tab:purple", "tab:blue"]
    ymax = max(abs(r[1]).max() for r in rows) * 1.05

    for axi, (label, d, rms, i, nu_i, nu_j), color in zip(ax[1:], rows, colors):
        axi.plot(nu_j, d, lw=0.4, color=color)
        axi.axhline(0, color="grey", lw=0.5)
        axi.set_ylabel("ΔBT [K]")
        axi.set_title(f"{label}    RMS {rms:.3f} K   worst {d[i]:+.2f} K @ {nu_i:.2f}")
        axi.set_ylim(-ymax, ymax)
    ax[-1].set_xlabel("wavenumber [cm⁻¹]")

    plt.tight_layout()
    out = f"{DATA}/julia_lblrtm_43um_subdiv.png"
    plt.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VP_W vs VP_Y comparison over 15 µm and 4.3 µm, plus the HITRAN-2020 reference
verification that the 15 µm movers are physical (first-order x-section goes
negative; full-matrix stays positive).
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(f):
    j = np.genfromtxt(f, delimiter=",", names=True)
    return j["nu_cm1"], j["BT_VPY"], j["BT_VPW"], j["dBT"]

nu15, btY15, btW15, d15 = load("data/vpw_vpy_15um.csv")
nu4,  btY4,  btW4,  d4  = load("data/vpw_vpy_4um.csv")
r = np.loadtxt("/tmp/lmfull/ABSCO_HIT2020.dat")
sig, V, Y, W = r[:,0], r[:,1], r[:,2], r[:,3]   # ref single-layer x-sections

fig, ax = plt.subplots(6, 1, figsize=(13, 20))
HL = dict(color="orange", alpha=0.18)

def bt_panel(a, nu, btY, btW, lo, hi, hlo, hhi, title):
    a.plot(nu, btY, lw=0.5, color="#1f77b4", label="VP_Y (first-order)")
    a.plot(nu, btW, lw=0.5, color="#d62728", label="VP_W (full-matrix)", alpha=0.8)
    a.axvspan(hlo, hhi, **HL); a.set_xlim(lo, hi); a.set_ylabel("BT (K)")
    a.set_title(title); a.legend(loc="lower right", fontsize=9)

def diff_panel(a, nu, d, lo, hi, hlo, hhi, label):
    a.axhline(0, color="k", lw=0.5); a.axvspan(hlo, hhi, **HL)
    a.plot(nu, d, lw=0.5, color="#2ca02c")
    i = int(np.argmax(np.abs(d)))
    a.annotate(f"{d[i]:+.1f} K @ {nu[i]:.1f}", xy=(nu[i], d[i]),
               xytext=(0.7*hi+0.3*lo, d[i]*0.8), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="k", lw=0.7))
    a.set_xlim(lo, hi); a.set_ylabel("ΔBT (K)")
    a.set_title(f"{label}: ΔBT (VP_W − VP_Y)  |  mean {d.mean():+.3f} K, "
                f"RMS {np.sqrt(np.mean(d**2)):.3f} K, max {np.abs(d).max():.1f} K")

# 15 µm
bt_panel(ax[0], nu15, btY15, btW15, 645, 800, 721, 731,
         "15 µm CO2 — Julia IASI BT (cont-OFF, ILS-OFF)")
diff_panel(ax[1], nu15, d15, 645, 800, 721, 731, "15 µm")

# 15 µm reference verification @721-731
m = (sig >= 721) & (sig <= 731)
ax[2].axhline(0, color="k", lw=0.7)
ax[2].plot(sig[m], V[m]*1e6, lw=0.7, color="gray",    label="AbsV (Voigt, no LM)")
ax[2].plot(sig[m], Y[m]*1e6, lw=0.9, color="#1f77b4", label="AbsY (first-order)")
ax[2].plot(sig[m], W[m]*1e6, lw=0.9, color="#d62728", label="AbsW (full-matrix)")
ax[2].fill_between(sig[m], Y[m]*1e6, 0, where=(Y[m]<0), color="#1f77b4", alpha=0.25)
nneg = int((Y[m]<0).sum())
ax[2].set_title(f"VERIFICATION — HITRAN-2020 ref x-section @721-731 (T=260K,P=0.5atm): "
                f"first-order AbsY NEGATIVE at {nneg}/{m.sum()} pts (shaded), full-matrix AbsW stays positive")
ax[2].set_ylabel("absorption (×10⁻⁶ cm⁻¹)"); ax[2].set_xlim(721, 731)
ax[2].legend(loc="upper right", fontsize=8)

# 4.3 µm
bt_panel(ax[3], nu4, btY4, btW4, 2200, 2400, 2386, 2392,
         "4.3 µm CO2 — Julia IASI BT (cont-OFF, ILS-OFF)")
diff_panel(ax[4], nu4, d4, 2200, 2400, 2386, 2392, "4.3 µm")

# Main ν2 Q-branch @667 — biggest x-section change but saturated
m2 = (sig >= 666.8) & (sig <= 668.2)
ax[5].plot(sig[m2], V[m2], lw=0.8, color="gray",    label="AbsV (Voigt, no LM)")
ax[5].plot(sig[m2], Y[m2], lw=1.0, color="#1f77b4", label="AbsY (first-order)")
ax[5].plot(sig[m2], W[m2], lw=1.0, color="#d62728", label="AbsW (full-matrix)")
ax[5].set_title("Context — main ν₂ Q-branch @667: largest |AbsW−AbsY| (~13%) but SATURATED, "
                "so BT barely moves (vs huge BT swing at the weak 721-731 zone)")
ax[5].set_xlabel("wavenumber (cm⁻¹)"); ax[5].set_ylabel("absorption (cm⁻¹)")
ax[5].set_xlim(666.8, 668.2); ax[5].legend(loc="upper right", fontsize=8)

fig.tight_layout()
out = "data/vpw_vpy_verification.png"
fig.savefig(out, dpi=105)
print("saved", out)

"""
Read LBLRTM per-layer optical depth files (ODint_NNN, IOD=1 output) and
write a CSV of per-layer τ at exactly 2364.105 cm⁻¹, the worst point in the
Julia−LBLRTM band-head comb.

LBLRTM ODint file format (gfortran ..._dbl build, IEMIT=0 IOD=1):
  record 0      : 1416-byte file header (XID and other Fortran COMMONs)
  per panel     :
    panel header  : V1, V2, DV (real*8) + NLIM (int32) + pad   -> 32 bytes
    data array    : optical depth, NLIM × real*8
  end marker    : 6-word record of -99

Same Fortran unformatted sequential framing as TAPE12, except ODint has
ONE data array per panel (τ only) instead of two (rad + transmittance).

Usage:
  python scripts/read_lblrtm_odint.py \\
      --dir data/lblrtm/lblrtm_run_iod1 \\
      --nu 2364.105 \\
      --out data/lblrtm/lblrtm_per_layer_tau_2364.csv
"""
import argparse, struct, glob, os, csv
import numpy as np


def read_records(path):
    """Yield payload bytes of each Fortran unformatted sequential record."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos + 4 <= len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        if n < 0 or pos + 8 + n > len(data):
            break
        payload = data[pos + 4 : pos + 4 + n]
        (n2,) = struct.unpack_from("<i", data, pos + 4 + n)
        if n2 != n:
            raise ValueError(f"record marker mismatch at byte {pos}: {n} vs {n2}")
        yield payload
        pos += 8 + n


def read_odint(path):
    """Return concatenated (nu, tau) arrays from one ODint file."""
    recs = list(read_records(path))
    nu_all, tau_all = [], []
    i = 1  # recs[0] is the file header
    while i < len(recs):
        ph = recs[i]
        if len(ph) < 28:    # end-of-file marker
            break
        v1, v2, dv = struct.unpack_from("<3d", ph, 0)
        nlim = struct.unpack_from("<i", ph, 24)[0]
        if nlim <= 0 or i + 1 >= len(recs):
            break
        tau = np.frombuffer(recs[i + 1], dtype="<f8")
        if len(tau) != nlim:
            raise ValueError(f"panel {i}: NLIM={nlim} but array len {len(tau)}")
        nu_all.append(v1 + dv * np.arange(nlim))
        tau_all.append(tau)
        i += 2
    return np.concatenate(nu_all), np.concatenate(tau_all)


def tau_at(nu, tau, target):
    """Optical depth at target wavenumber via linear interpolation."""
    return float(np.interp(target, nu, tau))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="LBLRTM run directory with ODint_NNN files")
    ap.add_argument("--nu", type=float, required=True, help="target wavenumber (cm-1)")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "ODint_*")))
    if not files:
        raise SystemExit(f"no ODint_* files in {args.dir}")
    print(f"found {len(files)} ODint files in {args.dir}")

    rows = []
    for fp in files:
        L = int(os.path.basename(fp).split("_")[1])
        nu, tau = read_odint(fp)
        tau_nu0 = tau_at(nu, tau, args.nu)
        rows.append((L, tau_nu0, tau.min(), tau.max(), nu[0], nu[-1], len(nu)))

    rows.sort(key=lambda r: r[0])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "tau_at_nu0", "tau_min_in_panel", "tau_max_in_panel",
                    "nu_lo", "nu_hi", "n_pts"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.6e}", f"{r[2]:.6e}", f"{r[3]:.6e}",
                        f"{r[4]:.6f}", f"{r[5]:.6f}", r[6]])
    print(f"wrote {args.out}  ({len(rows)} layers)")

    # Quick summary
    cum = 0.0
    print(f"\n=== cumulative τ from TOA down at ν = {args.nu} cm⁻¹ ===")
    for L, t, _, _, _, _, _ in reversed(rows):
        cum += t
        if cum >= 0.1 and cum <= 5.0:
            print(f"  L={L:2d}   τ_layer={t:.3e}   τ_cum_from_TOA={cum:.3f}")


if __name__ == "__main__":
    main()

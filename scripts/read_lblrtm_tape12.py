"""
Read an LBLRTM binary panel file (TAPE12) and write a CSV of
wavenumber, radiance, transmittance, and brightness temperature.

LBLRTM (gfortran ..._dbl build) writes panel files with BUFOUT =
`WRITE(unit) array`, i.e. Fortran *unformatted sequential* records framed
by INT32 byte-length markers (leading + trailing). Default REAL is promoted
to 8 bytes (-fdefault-real-8), so every numeric "word" is real*8.

File layout:
  record 0            : file header  (177 words; XID etc.)
  then per panel:
    panel header      : V1,V2,DV (real*8) + NLIM (int32) + pad  -> 32 bytes
    data array #1     : radiance,      NLIM * real*8
    data array #2     : transmittance, NLIM * real*8   (present when IEMIT=1)
  (terminated by an end record of 6 words = -99)

Radiance units are W/(cm^2 sr cm^-1). BT via inverse Planck.

Usage:
  python scripts/read_lblrtm_tape12.py <TAPE12> -o <out.csv>
"""

import struct, csv, argparse
import numpy as np

# Planck constants for radiance in W/(cm^2 sr cm^-1), nu in cm^-1
C1 = 1.191042972e-12   # 2 h c^2   [W cm^2 / sr]
C2 = 1.4387768788      # h c / k   [cm K]


def read_records(path):
    """Yield payload bytes of each Fortran unformatted sequential record."""
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    while pos + 4 <= len(data):
        (n,) = struct.unpack_from("<i", data, pos)
        if n < 0 or pos + 8 + n > len(data):
            break
        payload = data[pos + 4: pos + 4 + n]
        (n2,) = struct.unpack_from("<i", data, pos + 4 + n)
        if n2 != n:
            raise ValueError(f"record marker mismatch at byte {pos}: {n} vs {n2}")
        yield payload
        pos += 8 + n


def read_tape12(path):
    recs = list(read_records(path))
    nu_all, rad_all, tr_all = [], [], []
    i = 1                       # recs[0] is the file header
    while i + 2 < len(recs) + 1 and i < len(recs):
        ph = recs[i]
        if len(ph) < 28:        # end-of-file marker (6 words of -99)
            break
        v1, v2, dv = struct.unpack_from("<3d", ph, 0)
        nlim = struct.unpack_from("<i", ph, 24)[0]
        if nlim <= 0 or i + 2 >= len(recs):
            break
        rad = np.frombuffer(recs[i + 1], dtype="<f8")   # 1st array = radiance
        tr  = np.frombuffer(recs[i + 2], dtype="<f8")   # 2nd array = transmittance
        if len(rad) != nlim or len(tr) != nlim:
            raise ValueError(f"panel {i}: NLIM={nlim} but arrays "
                             f"{len(rad)},{len(tr)}")
        nu_all.append(v1 + dv * np.arange(nlim))
        rad_all.append(rad)
        tr_all.append(tr)
        i += 3
    nu = np.concatenate(nu_all)
    rad = np.concatenate(rad_all)
    tr = np.concatenate(tr_all)
    return nu, rad, tr


def planck_bt(nu, rad):
    rad = np.clip(rad, 1e-30, None)
    return C2 * nu / np.log1p(C1 * nu ** 3 / rad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tape12")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    nu, rad, tr = read_tape12(args.tape12)
    bt = planck_bt(nu, rad)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wavenumber_cm-1", "radiance_W_cm2_sr_cm", "transmittance", "BT_K"])
        for k in range(len(nu)):
            w.writerow([f"{nu[k]:.6f}", f"{rad[k]:.6e}",
                        f"{tr[k]:.6e}", f"{bt[k]:.4f}"])

    print(f"Read {args.tape12}: {len(nu)} points, {nu[0]:.4f}-{nu[-1]:.4f} cm-1")
    print(f"  radiance {rad.min():.3e}-{rad.max():.3e} W/(cm2 sr cm-1)")
    print(f"  BT {bt.min():.3f}-{bt.max():.3f} K (mean {bt.mean():.3f})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

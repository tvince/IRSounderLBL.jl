"""
Extract the MT-CKD CO2 continuum coefficient table from LBLRTM's contnm.f90.

The CO2 continuum (Hartmann line-coupling residual, isotopes 1+2) is NOT
distributed as a standalone dataset — even the canonical AER-RC/MT_CKD repo
keeps it only as Fortran `BLOCK DATA BFCO2` in src/contnm.f90 (only the H2O
continuum is externalized to netCDF). So the source file IS the upstream.

This parses the `DATA Fxxxx/ ... /` statements inside BLOCK DATA BFCO2,
concatenates them in COMMON-block order, and writes a CSV on the tabulated
grid. Asserts the LBLRTM-declared shape (V1=-4, DV=2, NPT=5003) so a parsing
slip can't pass silently.

The coefficient S(nu) has units (cm^3/mol)*1e-20; LBLRTM forms the absorption
as  k(nu) = S(nu) * n_CO2 * RHOAVE * 1e-20 * RADFN(nu,T).

Usage:
  python scripts/extract_ckd_co2.py \
      [--src ~/LBLRTM_build/LBLRTM/src/contnm.f90] \
      [-o data/mt_ckd_co2/mt_ckd_co2_coeffs.csv]
"""

import os, re, argparse

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.expanduser("~/LBLRTM_build/LBLRTM/src/contnm.f90")
DEFAULT_OUT = os.path.join(ROOT, "data", "mt_ckd_co2", "mt_ckd_co2_coeffs.csv")

# Declared in `DATA V1,V2,DV,NPT / -4.0, 10000.0, 2.0, 5003/`
EXPECT_V1  = -4.0
EXPECT_DV  = 2.0
EXPECT_NPT = 5003


def slice_block_data(text):
    """Return the text between BLOCK DATA BFCO2 and its END."""
    m = re.search(r"BLOCK DATA BFCO2(.*?)end block data BFCO2",
                  text, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("BLOCK DATA BFCO2 ... end block data BFCO2 not found")
    return m.group(1)


def strip_fortran(block):
    """Drop comment lines (leading '!') and continuation markers '&'."""
    out = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("!"):
            continue
        out.append(line)
    joined = "\n".join(out)
    # Remove continuation ampersands (line-end and line-start forms).
    joined = joined.replace("&", " ")
    return joined


def parse_grid(block):
    """Read V1,V2,DV,NPT from `DATA V1,V2,DV,NPT / ... /`."""
    m = re.search(r"DATA\s+V1,V2,DV,NPT\s*/(.*?)/", block,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("DATA V1,V2,DV,NPT statement not found")
    vals = [float(x) for x in m.group(1).replace("&", " ").split(",")]
    v1, v2, dv, npt = vals[0], vals[1], vals[2], int(vals[3])
    return v1, v2, dv, npt


def parse_f_arrays(block):
    """Find every `DATA Fxxxx/ ... /` array; return list of (label_int, values)."""
    arrays = []
    for m in re.finditer(r"DATA\s+F(\d+)\s*/(.*?)/", block,
                          re.IGNORECASE | re.DOTALL):
        label = int(m.group(1))
        body = m.group(2)
        vals = [float(tok) for tok in body.replace("&", " ").split(",")
                if tok.strip()]
        arrays.append((label, vals))
    return arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(os.path.expanduser(args.src)) as f:
        text = f.read()

    block = strip_fortran(slice_block_data(text))

    v1, v2, dv, npt = parse_grid(block)
    if (v1, dv, npt) != (EXPECT_V1, EXPECT_DV, EXPECT_NPT):
        raise AssertionError(
            f"grid mismatch: got V1={v1}, DV={dv}, NPT={npt}; "
            f"expected {EXPECT_V1}, {EXPECT_DV}, {EXPECT_NPT}")

    arrays = parse_f_arrays(block)
    # COMMON-block memory order == ascending label (F0000, F0001, F0051, ...).
    arrays.sort(key=lambda kv: kv[0])
    S = [v for _, vals in arrays for v in vals]

    if len(S) != npt:
        raise AssertionError(
            f"parsed {len(S)} coefficients from {len(arrays)} DATA arrays; "
            f"expected NPT={npt}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("nu_cm1,S_cm3_per_mol_x1e20\n")
        for i, s in enumerate(S):
            nu = v1 + dv * i
            f.write(f"{nu:.1f},{s:.6e}\n")

    nu_last = v1 + dv * (npt - 1)
    print(f"Wrote {args.out}")
    print(f"  {len(arrays)} DATA arrays -> {len(S)} coefficients")
    print(f"  grid: V1={v1} DV={dv} NPT={npt}  ({v1:.1f} to {nu_last:.1f} cm-1)")
    # Quick window peek (CO2 nu2 wing where the 0.44 K residual lives).
    for target in (667.0, 700.0, 714.0, 740.0, 790.0):
        i = round((target - v1) / dv)
        print(f"  S({v1 + dv*i:.1f}) = {S[i]:.4e}")


if __name__ == "__main__":
    main()

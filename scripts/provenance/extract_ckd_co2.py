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

It also parses the two corrections LBLRTM applies on top of the raw S table:
  - XFACCO2: 500 multiplicative factors for 2000-2998 cm-1 (mt_ckd_2.5), every
    2 cm-1. Applied as  fco2(j) = cfac * fco2(j)  in the main loop.
  - tdep_bandhead: 25 exponents for 2386-2434 cm-1, giving a temperature
    correction  tcor = (T/t_eff)**tdep_bandhead  (t_eff=246 K) in FRNCO2.
Both default to identity (xfac=1, tdep=0) outside their ranges, so the full
absorption is:
    k(nu) = S(nu) * xfac(nu) * (T/t_eff)**tdep(nu)
            * n_CO2 * RHOAVE * 1e-20 * RADFN(nu,T).

Output CSV columns: nu_cm1, S_cm3_per_mol_x1e20, xfac, tdep_exp.

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

# XFACCO2: 500 factors on 2000..2998 cm-1 (JFAC=(VJ-1998)/2, 1-based).
XFAC_N      = 500
XFAC_V1     = 2000.0
# tdep_bandhead: 25 exponents at S-array indices 1196..1220 (1-based),
# i.e. nu = V1 + DV*(i-1) = 2386..2434 cm-1.
TDEP_I_LO   = 1196
TDEP_I_HI   = 1220
EXPECT_TEFF = 246.0


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


def parse_xfacco2(text):
    """Read the 500 XFACCO2 correction factors (main subroutine, not BLOCK DATA)."""
    m = re.search(r"DATA\s+XFACCO2\s*/(.*?)/", text,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("DATA XFACCO2 statement not found")
    vals = [float(tok) for tok in m.group(1).replace("&", " ").split(",")
            if tok.strip()]
    if len(vals) != XFAC_N:
        raise AssertionError(f"XFACCO2: parsed {len(vals)} values, expected {XFAC_N}")
    return vals


def parse_tdep_bandhead(text):
    """Read the 25 tdep_bandhead exponents and t_eff (FRNCO2 subroutine)."""
    m = re.search(r"tdep_bandhead\(i\),\s*i=\d+,\d+\)\s*/(.*?)/", text,
                  re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError("tdep_bandhead implied-do DATA statement not found")
    vals = [float(tok) for tok in m.group(1).replace("&", " ").split(",")
            if tok.strip()]
    n_expect = TDEP_I_HI - TDEP_I_LO + 1
    if len(vals) != n_expect:
        raise AssertionError(f"tdep_bandhead: parsed {len(vals)}, expected {n_expect}")

    mt = re.search(r"DATA\s+t_eff\s*/\s*([\d.]+)", text, re.IGNORECASE)
    if not mt:
        raise ValueError("DATA t_eff statement not found")
    t_eff = float(mt.group(1).rstrip("."))
    if t_eff != EXPECT_TEFF:
        raise AssertionError(f"t_eff: got {t_eff}, expected {EXPECT_TEFF}")
    return vals, t_eff


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

    # ── Corrections, aligned onto the S grid (identity by default) ──────────
    full = strip_fortran(text)
    xfacco2 = parse_xfacco2(full)
    tdep_bh, t_eff = parse_tdep_bandhead(full)

    xfac = [1.0] * npt          # multiplicative; 1 outside 2000-2998
    tdep = [0.0] * npt          # exponent; 0 outside 2386-2434

    # XFACCO2[k] (k=1..500) -> nu = 1998 + 2k -> grid row r = (nu - V1)/DV.
    for k, fac in enumerate(xfacco2, start=1):
        nu = XFAC_V1 + 2.0 * (k - 1)
        r = round((nu - v1) / dv)
        xfac[r] = fac
    # tdep_bandhead is indexed by the 1-based S index i (1196..1220).
    for off, exp in enumerate(tdep_bh):
        i = TDEP_I_LO + off          # 1-based S index
        tdep[i - 1] = exp

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("nu_cm1,S_cm3_per_mol_x1e20,xfac,tdep_exp\n")
        for i, s in enumerate(S):
            nu = v1 + dv * i
            f.write(f"{nu:.1f},{s:.6e},{xfac[i]:.6f},{tdep[i]:.6e}\n")

    nu_last = v1 + dv * (npt - 1)
    print(f"Wrote {args.out}")
    print(f"  {len(arrays)} DATA arrays -> {len(S)} coefficients")
    print(f"  grid: V1={v1} DV={dv} NPT={npt}  ({v1:.1f} to {nu_last:.1f} cm-1)")
    print(f"  XFACCO2: {len(xfacco2)} factors on {XFAC_V1:.0f}-"
          f"{XFAC_V1 + 2*(XFAC_N-1):.0f} cm-1")
    print(f"  tdep_bandhead: {len(tdep_bh)} exponents, t_eff={t_eff} K "
          f"(nu {v1 + dv*(TDEP_I_LO-1):.0f}-{v1 + dv*(TDEP_I_HI-1):.0f})")
    # Window peeks: 15 um (corrections inert) and 4.3 um (corrections active).
    for target in (667.0, 714.0, 790.0, 2200.0, 2400.0, 2500.0):
        i = round((target - v1) / dv)
        print(f"  nu={v1 + dv*i:7.1f}  S={S[i]:.4e}  xfac={xfac[i]:.4f}  "
              f"tdep={tdep[i]:.4f}")


if __name__ == "__main__":
    main()

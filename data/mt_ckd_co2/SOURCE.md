# MT-CKD CO₂ continuum — provenance and AER license

`mt_ckd_co2_coeffs.csv` holds the CO₂ line-coupling continuum coefficients
(`nu_cm1, S_cm3_per_mol_x1e20, xfac, tdep_exp`) used by `co2_continuum` in
`src/CrossSections/continuum.jl`. The table is **derived** from AER's LBLRTM
source — parsed from `contnm.f90` `BLOCK DATA BFCO2` (5003 points, V1=−4, DV=2;
byte-identical to the AER-RC/LBLRTM master), and it also carries the XFACCO2
(2000–2998 cm⁻¹) and bandhead-temperature (2386–2434 cm⁻¹) corrections. See
`scripts/provenance/extract_ckd_co2.py`.

This is third-party data. It is **not** covered by the package's MIT license and
carries AER's own terms, reproduced verbatim below as that license requires.

## AER copyright notice (reproduced per the license)

```
 Copyright ©, Atmospheric and Environmental Research, Inc.

 All rights reserved. This source code was developed as part of the
 LBLRTM software and is designed for scientific and research purposes.
 Atmospheric and Environmental Research Inc. (AER) grants USER the right
 to download, install, use and copy this software for scientific and
 research purposes only. This software may be redistributed as long as
 this copyright notice is reproduced on any copy made and appropriate
 acknowledgment is given to AER. This software or any modified version
 of this software may not be incorporated into proprietary software or
 commercial software offered for sale without the express written
 consent of AER.

 This software is provided as is without any express or implied warranties.
```

Questions to AER: aer_contnm@aer.com

## Acknowledgment / citation

CO₂ continuum from AER's MT_CKD / LBLRTM. Please cite:

> Mlawer, E. J., Payne, V. H., Moncet, J.-L., Delamere, J. S., Alvarado, M. J.,
> & Tobin, D. C. (2012). *Development and recent evaluation of the MT_CKD model
> of continuum absorption.* Phil. Trans. R. Soc. A, 370, 2520–2556.
> https://doi.org/10.1098/rsta.2011.0295

Upstream: AER-RC/LBLRTM (https://github.com/AER-RC/LBLRTM).

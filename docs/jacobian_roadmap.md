# Jacobian & Estimation Roadmap

**Status:** design / not yet implemented (drafted 2026-06-12; corrections from a
code audit folded in 2026-06-13 — see §2.1, §2.2, §3, §6.1, §6.4)
**Goal:** add analytic Jacobians (weighting functions) to the validated forward
model so it can drive estimation/retrieval methods (optimal estimation,
Gauss–Newton, Levenberg–Marquardt).

The forward model is mature and validated (CO₂ LBL + continuum ~0.05 K vs
LBLRTM; CO₂ line mixing validated vs the HITRAN reference and speed-optimized).
It currently runs *forward only*. Jacobians are the missing half that turn a
forward model into a sounder you can invert.

---

## 1. What we need

For estimation we need the **Jacobian / weighting-function matrix**

```
K = ∂y/∂x        size: n_channels × n_state
```

- **y** — measurement vector: channel brightness temperature (or radiance).
- **x** — state vector, drawn from:
  - temperature profile `T(z)`            (per level)
  - species VMR profiles `VMR_s(z)`       (per level, per retrieved species)
  - surface temperature `T_sfc`
  - surface emissivity `ε`
  - (later, optionally) instrument / line-shape nuisance parameters

`K` is consumed by the estimation layer (Section 5).

---

## 2. Architecture: exploit the RT factorization

The forward chain factorizes, and **each factor has a different best
differentiation strategy**. We do *not* brute-force whole-model AD.

```
x (state)                          best derivative strategy
 │  T(z), VMR_s(z), T_sfc, ε
 ▼
layer optical depth                τ_i = Σ_s n_{s,i}·σ_{s,i}(T_i,p_i)·Δz_i
 │   ∂τ/∂VMR_s  → LINEAR (n ∝ VMR)              → analytic, ~free
 │   ∂τ/∂T_i    → needs ∂σ/∂T (the hard one)   → analytic or localized fwd-AD
 ▼
Schwarzschild + CIM RTE            → monochromatic radiance L(ν)
 │   ∂L/∂τ_i, ∂L/∂T_i, ∂L/∂T_sfc, ∂L/∂ε  → ANALYTIC & cheap
 ▼
ILS convolution                    → channel radiance → BT
     ∂(channel)/∂(mono) = ILS matrix          → LINEAR, exact (matmul)
```

### Chosen strategy: **semi-analytic**

| State element | Strategy | Why |
|---|---|---|
| VMR_s(z) | **Analytic** | VMR enters via number density. For a clean trace species (no self-broadening, no continuum) `∂τ_s/∂VMR_s = τ_s/VMR_s` exactly and free. **H₂O and CO₂ carry extra terms** — see §2.1. |
| T_sfc, ε | **Analytic** | Appear explicitly in the RTE surface boundary term. |
| RTE w.r.t. {τ_i}, {T_i} | **Analytic** | Schwarzschild + CIM source has closed-form derivatives. |
| mono → channel | **Linear operators + diagonal** | Two linear ops (`apply_ils` convolution, then `_resample_to_iasi`) on the mono Jacobian, then a diagonal `∂BT/∂R`. See §2.2. |
| T(z) (via ∂σ/∂T) | **Analytic** (preferred) *or* localized forward-mode AD | The one hard derivative; see §3. |

### 2.1 The VMR Jacobian is *not* uniformly free/exact

`∂τ_s/∂VMR_s = τ_s/VMR_s` holds **only** for an optically-thin trace species with
no self-broadening and no continuum contribution. The code (`iasi_forward_model`,
`compute_voigt_cross_sections`) shows two ways this leaks for the species you most
want to retrieve:

- **H₂O:** `vmr_self = vmr` feeds `pressure_broadened_width` (self-broadening) and
  `pressure_shift`, so the cross section `σ` *itself* depends on VMR — the linearity
  in number density breaks and an extra `∂σ/∂VMR_self` term appears.
- **Continuum / CIA:** these are summed into the same `τ_layers` and are **quadratic**
  in VMR (MT-CKD self-continuum ∝ vmr²; CO₂/N₂/O₂ CIA ∝ density²). So for H₂O and CO₂,
  `∂τ/∂VMR` carries continuum terms of order `2·self + foreign`, not `τ_cont/VMR`.

All of these terms are still analytic — Phase 2 is not harder — but "nearly free,
exact" applies to clean trace gases, **not** to H₂O/CO₂. Validating Phase 2 against
FD for H₂O/CO₂ without these terms will look like a bug. Build them in from the start.

### 2.2 The "linear tail" is two operators plus a diagonal

The mono→channel map is not a single ILS matmul. It is `apply_ils` (convolution,
linear) **then** `_resample_to_iasi` (interpolation, linear) **then**
`brightness_temperature` (`∂BT/∂R`, a per-channel diagonal = inverse-Planck slope).
So `K_BT = diag(∂BT/∂R) · Resample · ILS · K_radiance(mono)`. All three are cheap and
exact; just compose them explicitly so the channel-space assembly is unambiguous.

### Why not whole-model reverse-mode AD (Zygote / Enzyme)

The forward model is heavily **in-place / mutating** for speed (per-layer τ
loops, the vectorized Faddeeva kernel). That is exactly the code reverse-mode AD
tools struggle with: Zygote rejects mutation; Enzyme handles it but is brittle on
complex numerical kernels. The RT structure is well understood and factorizes
cleanly, so analytic derivatives win on **both speed and maintainability**.
Forward-mode AD is kept as a *scalpel* for one localized derivative, not dragged
through the whole model.

---

## 3. The temperature Jacobian (the hard piece)

`∂BT/∂T_i` is the only derivative needing real work, because layer temperature
enters in three places:

1. **Cross section** `∂σ/∂T` — line strength `S(T)` (partition-function ratio
   `Q(T_ref)/Q(T)` + Boltzmann factor + stimulated-emission term) and line
   widths (Doppler `γ_D ∝ √T`, Lorentz `γ_L ∝ (T₀/T)^n · p`). Line mixing adds
   the T-dependence of `Y(T)` / the relaxation matrix.
2. **Number density** `∂n/∂T` — ideal gas, `n ∝ p/T`.
3. **Source term** `∂B/∂T` — Planck function, and the **CIM source function's**
   own T-dependence (mass-weighted T, Padé form) — must be differentiated
   *consistently* with the forward CIM, not approximated separately.

**Recommendation: hand-analytic `∂σ/∂T` is the primary route; ForwardDiff is the
fallback.** This reverses an earlier lean toward forward-mode AD — an audit of the
code shows the AD route is more obstructed, and the analytic route easier, than it
first appears.

*Why analytic is easier than it looks.* `σ = Σ_j Snorm_j · H(x_j, y_j)`, and the
Faddeeva function has a closed-form derivative, `dw/dz = −2z·w(z) + 2i/√π`, which
gives both `∂H/∂x` and `∂H/∂y` while **reusing the same `erfcx` evaluation already
in `faddeeva_voigt`**. With `x, y, Snorm` all simple functions of `S(T)`, `γ_L(T)`,
`γ_D(T)`, the chain rule for `∂σ/∂T` is mechanical and cheap.

*Why ForwardDiff is obstructed (audit before relying on it — §6.3).* Duals would
have to flow through, and the current code blocks them at three points:
- `temperature_scaled_intensity(line, T::Float64)` and
  `pressure_broadened_width(..., T::Float64)` have **hardcoded `::Float64`
  signatures** (Duals won't even dispatch).
- `pressure_broadened_width` ends with `return Float64(γ_L), Float64(γ_D)` — that
  cast **silently strips the derivative**: a wrong answer that looks plausible, not
  an error.
- σ is assembled by a **`KernelAbstractions.@kernel` writing preallocated `Float64`
  arrays**, not a scalar function — "AD on just the kernel" means writing a parallel
  scalar Dual path anyway. (`erfcx` needing a Dual rule is the *least* of it.)

Hand-derive `∂n/∂T` and `∂B/∂T` either way (both trivial).

**Accuracy floor — partition-function table quantization.** `Q_ratio` interpolates
the **TIPS-2024 table at 1 K resolution** (`_tips_lookup` uses `floor(Int, T)`), so
`∂Q/∂T` — by *any* method (analytic, AD, or a tight FD step) — is a piecewise-constant
staircase, the secant slope of a 1 K-spaced table. This, not the differentiation
method, sets the accuracy floor on the `S(T)` part of the temperature Jacobian. Plan
to either fit a smooth `Q(T)` or take a multi-K central difference of the table for
the derivative path; validate ∂S/∂T against an FD step large enough to span the
quantization.

---

## 4. Required refactor: one-pass linearization context

The forward model must **emit its intermediates** so Jacobians assemble from
cached state instead of recomputation:

- per-layer cross sections `σ_{s,i}`
- per-layer optical depths `τ_i` (and per-species contributions)
- per-layer transmittances and source terms

Plan: `iasi_forward_model` gains an option to return a **linearization context**
(the internal state) alongside the radiance. Jacobians are then assembled from
that context + the analytic RTE/ILS derivatives in roughly **1–2 forward-model
costs**, not `n_state` costs.

This is where the **deferred LM per-(band,T) eigenmode caching** finally pays off
— a retrieval loop does many forward evaluations at nearby states, so caching the
expensive line-mixing eigendecompositions amortizes across them.

---

## 5. Phased plan (each phase validated against finite differences)

> **Validation discipline:** finite differences are the ground truth. Build the
> FD harness *first* and validate every analytic/AD Jacobian against it — the
> same discipline used for LBLRTM/ARTS forward-model validation.

- **Phase 0 — Design + FD reference harness.**
  Define the state-vector abstraction and the `K`-matrix output type. Implement a
  finite-difference Jacobian (perturb each state element, re-run). Slow but exact-
  enough; it is the reference for all later phases. Decide the cutoff-freezing
  policy (§6.1) here.

- **Phase 1 — Analytic RTE Jacobian.**
  Differentiate the Schwarzschild + CIM RTE w.r.t. layer optical depths, layer/
  surface temperatures, and emissivity, holding cross sections fixed. Validate vs
  FD. This is the chain backbone.

- **Phase 2 — VMR Jacobians.**
  Exploit number-density linearity (`∂τ_s/∂VMR_s`), combine with Phase 1 → full
  `∂BT/∂VMR_s(z)` per species per level. Validate vs FD. **First genuinely useful
  retrieval Jacobian** (trace-gas retrieval).

- **Phase 3 — Temperature Jacobian.**
  Implement `∂σ/∂T` (hand-analytic primary, ForwardDiff fallback — §3) + `∂n/∂T` +
  `∂B/∂T` (incl. CIM consistency). Mind the partition-table quantization floor (§3)
  and keep VP_W LM FD-only at first (§6.4). Validate vs FD. Labor- and
  accuracy-critical phase.

- **Phase 4 — Performance / caching.**
  Refactor the forward model to emit the one-pass linearization context (§4);
  wire in LM eigenmode caching. Target: full `K` in ~1–2 forward-model costs.

- **Phase 5 — Estimation layer.**
  Build the inverse on top of the `K`-matrix API: optimal estimation /
  Gauss–Newton / Levenberg–Marquardt with a-priori and measurement covariances
  and iteration. This is the end goal — Jacobians *enabling estimation*.

---

## 6. Risks & decisions to settle early

### 6.1 Lossy approximations break Jacobian consistency
State-dependent active sets and clamps drop or floor terms based on the *current*
state. Under perturbation a line/band/value can cross a threshold, producing
derivative **discontinuities** — FD noise *and* analytic/FD mismatch. The
discontinuity sources in the current code:
- **`dptmn` weak-line rejection** (`_reject_weak_lines`) — per-layer, keyed on T and
  VMR via `coef`; lines enter/leave as the state moves.
- **band-strength cutoff** (`min_band_strength`, "#4") on the LM path.
- **the `max(acc, 0.0)` clamp** in `voigt_cross_section_kernel!` — floors σ at zero,
  a kink wherever σ crosses zero (only active in deep windows, but FD will see it).

Likely fix: **freeze the active set / clamp masks across the linearization** (compute
them once at the linearization point and hold them fixed for the forward pass and all
Jacobian columns), or disable the cutoffs during Jacobian assembly. Decide in Phase 0.

### 6.2 CIM source-function consistency
The CIM source (mass-weighted T, Padé form) has its own T-dependence. Its
Jacobian (Phases 1 & 3) must be derived consistently with the forward CIM — do
not let the source approximation and its derivative drift apart.

### 6.3 ForwardDiff compatibility of the cross-section kernel
Localized forward-AD on `σ(T)` requires the kernel to admit `Dual` numbers:
no hardcoded `Float64` element types in the hot path, and a derivative rule (or
generic implementation) for `erfcx`. Audit this before committing to the AD route
in Phase 3; fall back to analytic `∂σ/∂T` if the kernel resists.

### 6.4 VP_W line-mixing temperature derivative is the hardest atom
`∂σ/∂T` with full-matrix line mixing (VP_W) active requires differentiating an
**eigendecomposition of the relaxation matrix** (eigenvector perturbation theory) —
materially harder than the first-order VP_Y `Y(T)` term. Do **not** block Phase 3 on
it: keep the VP_W temperature derivative **FD-only initially** (or restrict the
analytic T-Jacobian to the no-LM / VP_Y paths) and revisit once the rest of Phase 3
validates. The hand-analytic `∂σ/∂T` plan in §3 covers the plain-Voigt and VP_Y cases.

### 6.5 State-vector & units conventions
Fix early: VMR vs log-VMR state (log is standard for positive-definite trace
gases and linearises the multiplicative response), temperature in K, layer vs
level quantities, and the ordering/blocking of `x` and `K`. The estimation layer
(Phase 5) depends on these being stable.

---

## 7. Open questions for the estimation layer (Phase 5)
- Which inverse method first — optimal estimation (Rodgers) vs plain Gauss–Newton?
- A-priori covariance construction (correlation length scales per species/T).
- Measurement covariance / channel selection.
- Convergence + error characterization (averaging kernels, retrieval covariance).

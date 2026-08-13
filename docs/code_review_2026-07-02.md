# IRSounderLBL.jl — code review & optimization critique (2026-07-02)

Scope: full-source review ahead of JuliaHub publication, with emphasis on the
Jacobian and line-mixing hot paths. Context: LBL + continuum validated vs LBLRTM
to ~0.05 K; analytic Jacobian FD-exact (~1e-7) incl. continuum and LM coupling;
Voigt grad kernel vectorized (commit 52cba83, retrieval ~10 min/pixel); VP_Y
retrieval experiment killed at ~24 min/iter — LM Jacobian is the bottleneck.

## Overall state

The physics core is in genuinely good shape. Docstrings are unusually thorough —
they explain the math, the conventions, and the validation history. The main gaps
are (a) the line-mixing Jacobian hot path, which is architecturally unoptimized in
a way the Voigt path no longer is, and (b) packaging hygiene, where several items
would block a JuliaHub release.

## Line mixing: the big remaining performance lever

Measured: VP_Y retrieval ~24 min for one iteration vs the whole Voigt retrieval in
~10 min. The cost is structural, and there is more headroom than the two fixes
already planned:

1. **The Y(T) computation does O(n²) work with dictionary lookups inside the
   double loop — per band, per layer, per perturbation call.**
   `_build_W_matrix` (line_mixing.jl:447) runs `sortperm`, allocates an n×n
   matrix, and inside the i,j loop does a `Symbol` construction +
   `getfield(tbl, sub)` + `Dict` lookup per pair. All of the pair structure —
   which (i,j) couple, which sub-table, the (W0R, B0R) coefficients, the symmetry
   skips — is **T-independent**. Precompute it once at load time (or lazily per
   band) as a flat sparse list of `(i, j, W0R, B0R)` tuples; each temperature
   evaluation is then just `exp(W0R − B0R·log(T0/T))` over that list. Benefits
   the *forward* model too, not just the Jacobian.

2. **Better still: tabulate Y(T) per band and interpolate.** Y is smooth in T
   (products of power laws and exponentials); the original HITRAN Fortran package
   itself fits over a handful of reference temperatures. A per-band Y table on a
   10–20-point grid over 150–330 K, built once when `VPYLineMixing` is
   constructed, removes the O(n²) rebuild from the hot path entirely — for all
   layers, all iterations, and all FD perturbation calls. It also gives an
   *analytic* ∂Y/∂T from the interpolant (feeds item 4).

3. **The two already-planned fixes remain valid and stack with the above:**
   - cache Y at the unperturbed T — the centre and both ∂p calls in
     `_species_cross_section_grad` (cross_section_jacobian.jl:281-287) rebuild
     identical Y three times per layer;
   - a `need_p` gate — a T(p)+T_sfc state never consumes ∂Δσ/∂p, so 2 of the 5
     perturbation calls are dead work (`analytic_jacobian` only reads `gr.dp`
     when `do_coupling && is_retr`).

4. **Longer term: make the VP_Y derivative analytic instead of central-FD.** The
   dispersive term is `Σ Y·p·S(T)·f·Im[w(z)]` — structurally identical to the
   Voigt sum with Im w in place of Re w, and the full w-gradient
   (`w′ = −2z·w + 2i/√π`) is already computed in `voigt_grad_kernel!`. With
   ∂Y/∂T from the table, ∂Δσ/∂T and ∂Δσ/∂p are closed-form. That collapses 5
   perturbation evaluations to 1 and lets the dispersive lines fold into the same
   KA kernel pass.

5. **The dispersive evaluator has GC and threading inefficiencies independent of
   Y.** `_lm_band_dispersive` allocates a full-grid `zeros(n_ν)` per band and the
   caller does `Δσ .+= ...` — with 339 bands and a ~62k-point grid that is
   ~170 MB of transient allocation per perturbation call, ×5 per layer, ×~50
   layers. It also launches `Threads.@threads` 339 times per call on mostly-small
   ν-windows. Clean fix mirrors the Voigt grad port: concatenate all bands'
   active lines into one sorted-by-ν₀ array with per-line `YSnorm`, and evaluate
   in a single windowed KA kernel launch (binary-search
   `_lower_bound`/`_upper_bound`, same as `voigt_cross_section_kernel!`). One
   launch, in-place accumulation, no per-band vectors.

Together, 1+2+5 should bring the VP_Y forward evaluation close to plain-Voigt
cost; 3+4 bring the LM Jacobian to parity with the analytic Voigt Jacobian.

**Recommended sequence:** sparse coupling table (1) → Y cache + need_p gate
(3, the already-planned resume) → single-kernel dispersive (5) → Y(T) table +
analytic derivative (2+4) only if profiling still says so.

**VP_W notes:** `band_modes` computes `inv(X)` explicitly (line_mixing.jl:792);
`X \ (ρ .* d_eff)` via the factorization is cheaper and better conditioned.
`default_vpw_whitelist`'s internal `band_S` closure duplicates
`_band_eff_strength` verbatim — delete one.

## Jacobian path (Voigt side)

In good shape after the kernel port. Remaining, all modest:

- **Duplicate storage:** when both `need_T` and coupling are active,
  `dτdTcg[sp][:,k]` and `dτdTc[sp][:,k]` store the identical `gr.dT .* coef`
  (vmr_jacobian.jl:256, 265) — two ~24 MB matrices per species holding the same
  numbers. Alias them.
- **`need_p`/`need_self` compile-time gates** in `voigt_grad_kernel!` (via `Val`)
  would trim the multiply-accumulate tail for T-only retrievals, but the erfcx
  dominates — only worth it if a profile shows it.
- **`Sa_inv = inv(Sa)`** in `optimal_estimation` and the recomputed
  `Se_fac \ resid` for `chi2` are minor; a Cholesky of Sa would be marginally
  cleaner for ill-conditioned Matérn priors with small length scales.
- The RTE Jacobian, ILS-FFT reuse, and Se prefactoring are all sound — nothing
  structural to improve.

## Correctness observations (nothing alarming)

- `_lm_band_dispersive` leaves `ν0_b/f_b/y_b` **uninitialized** for lines skipped
  by `S_T ≤ 0` and relies on the `YSnorm[k] == 0.0` guard to never read them.
  Correct today, but fragile — a future edit that reorders the guards reads
  garbage. Initialize `ν0_b` to `NaN` or restructure to compact arrays of active
  lines (the compact form is what the single-kernel refactor wants anyway).
- The `max(·,0)` clamp discipline (zeroing derivatives where σ clamps) is
  consistently applied in both the kernel and the LM wrapper — good.
- `Nair_per_vmr = 2.1209e22` is hardcoded in four places (optical_depth.jl ×2,
  vmr_jacobian.jl:198, iasi.jl:193). Hoist to one `const` — a future unit fix
  must currently hit all four.

## JuliaHub publication blockers

In rough priority order:

1. **The UUID is a placeholder** (`a1b2c3d4-e5f6-...`). The General registry
   requires a real UUID4; regenerate with `UUIDs.uuid4()` before anything else
   references it.
2. **HITRAN data licensing** — the repo commits HITRAN-derived CIA/linelist
   tables with no redistribution license. These must move behind
   `download_data()`/Artifacts before the repo goes public, which registration
   implies.
3. **Dependency bloat.** `CUDA` is a hard dep but only referenced via
   `isdefined(Main, :CUDA)` in strategy.jl — drop it or make it a weakdep with an
   extension. `Metal` is `using`'d unconditionally in voigt.jl and belongs in a
   package extension (it drags GPU artifacts onto every Linux CI/user install).
   `Plots` should not be a dependency of a physics library at all — no
   `using Plots` in src, likely vestigial; same check for `Tullio` and `Unitful`
   (neither appears in src). Every dropped dep cuts install time and precompile
   surface substantially.
4. **API generality vs the stated scope.** The pitch is "nadir-viewing
   hyperspectral satellites," but the public API is IASI-shaped:
   `iasi_forward_model`, `IASIInstrument`, `iasi_grid`, `read_iasi_l1c`. The
   struct is already generic (ν range, Δν, OPD, FWHM); consider a generic name
   (`FTSInstrument` / `forward_model`) with IASI/CrIS/IASI-NG constructors,
   keeping `iasi_*` as thin aliases. Renaming after registration is much more
   painful.
5. **Docs and QA scaffolding:** no Documenter build (the docstrings are excellent
   raw material, so this is mostly wiring). Add Aqua.jl (catches stale deps,
   missing compat, ambiguities) and ideally JET to the test suite. CI exists.

## One-sentence summary

The science and the Voigt/Jacobian machinery are publication-quality; the
line-mixing evaluator is the last subsystem still built as "correct first, fast
later," with a well-defined fix path (T-independent coupling tables →
cached/tabulated Y → one windowed kernel); and Project.toml needs real surgery
(UUID, weakdeps, data licensing) before this can safely touch a registry.

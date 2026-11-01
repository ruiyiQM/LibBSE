# LRI kernel regression fixtures

Author: Ruiyi Zhou.

These are fixed-input correctness checks. They do not impose wall-time or
speedup thresholds on a shared CI runner. Larger Si performance measurements
remain separate Euler benchmarks.

`inputs.tar.gz` is the previously validated `hchain_coarse4_fine8` input export:
a dimerized, two-H periodic chain (z bonds 1.0/1.5 Å, transverse cell 12 Å),
coarse 1×1×4 RI/Wc with dense 1×1×8 KS/QP data, 1 occupied and 4 virtual bands
(32 transitions). It contains generated FHI-aims electronic data and LibRPA Wc,
not FHI-aims source or an executable. It is copied unchanged from the earlier
local/Euler `extra-fixtures.tar.gz`, SHA256
`733f2d5f348d858ba2e94c23bad69981937b9c57f9a95ee3d638efc31111f9a5`.
`archives.txt` pins the archive; the runner verifies it before extraction and
rejects absolute paths, parent traversal and non-regular members. CI does not
run a producer or regenerate input/reference data.

The Si cases reuse the existing [Si 222 inputs](../si222/README.md), their
checksums and lossless screening restoration. No Si data is duplicated.

| Case | Coverage |
| --- | --- |
| `h_cached` | 8 MPI × 2 OpenMP, cached contraction; actual TR reuse is required |
| `h_cached_capped` | Request 4 threads, cap to 2; cached packet rows remain correct |
| `h_bounded` | Zero cache budget takes the bounded route |
| `h_mixed_capped` | Alternating owners have zero budget; collective fallback, capped team |
| `h_complex` | W rotated by 0.8+0.3i; arbitrary complex W and actual TR reuse |
| `h_owner_fallback` | 32 MPI, one k per owner; too few TR orbits triggers cyclic ownership |
| `si_cached` | Real Si auxiliary/atomic dimensions and cached Hx |
| `si_small_budget` | Positive 1 MiB budget is too small; size-based fallback |

Each run uses `test_ri_streaming` to compare all four A/B contributions (12
checks), three complex probes and singlet/RPA/triplet coefficients (27 streamed
and 9 AO/R Hx checks), plus 6 adjoint checks for Hermitian input. FFTW-enabled
builds add 9 FFT comparisons; disabled builds require their absence. Synthetic
complex W intentionally breaks Hermiticity, so only its adjoint checks are
omitted. Every numerical error must be finite and ≤5×10⁻¹² Ry.

The gate also requires every rank's partition and all six screened-Hx records,
the expected cache/fallback route, valid TR acceptance residuals and the guards
against using Cartesian reference methods on a distributed-k object. H-chain
cached/complex cases must report at least one accepted TR pair. These explicit
Hx comparisons are independent of Haydock quadrature or optical normalization;
the separate Si spectrum regressions cover those.

```sh
ctest --test-dir build -L lri_kernel --output-on-failure
```

Reports, commands/executable hashes and per-rank logs are kept in
`build/lri_kernel_runs/` and uploaded by CI on failure as well as success.

# LibBSE unit tests

The tests are intentionally small and deterministic. Run all of them with:

```sh
ctest --test-dir build-bse --output-on-failure
```

## `test_parameter.cpp`

- `require()` turns a failed Boolean test condition into a descriptive test
  failure.
- `test_whitespace_separated_parameters()` checks whitespace-separated input,
  all core BSE parameter names, the multi-value `bse_spin_types` list,
  relative `input_dir`/`output_dir` resolution, and `bse_tda=full` dispatch.
- `test_librpa_style_assignments_and_last_value_wins()` checks optional
  `key = value` syntax, inline comments, Boolean parsing, and the LibRPA/ABACUS
  rule that the last occurrence of a parameter wins, plus the default
  `output_dir=libbse.d`.
- `test_coarse_kgrid_mode()` checks that `bse_use_fine_kgrid=0` selects the
  SCF coarse-grid calculation path.
- The whitespace-separated test also checks the producer, QP source,
  screened-data, spectrum, and wavefunction-gauge controls used by the
  FHI-aims path.
- Haydock polarization parsing checks the `xyz` default, case normalization,
  and rejection of duplicate or invalid directions. The real-Si
  `LibBSE_real_Si222_haydock_native_w_x` regression compares the two-column
  x spectrum against frozen ELPA and checks that only the x chain is saved.
- Haydock quadrature parsing preserves the `gauss` default, accepts `gagq`
  case-insensitively, and rejects invalid or unused GAGQ settings. The Haydock
  unit test checks degree-2k moment matching without extra Hx, exhausted and
  short chains, malformed coefficients, and a nonpositive GAGQ extension.
- `test_invalid_or_unsupported_parameters_are_rejected()` checks rejection of
  length gauge, unsupported or duplicate spin types, invalid pure-IPA and
  k-grid modes, unknown or obsolete keywords, and malformed integer values.
- `main()` runs the parameter tests and reports exceptions as a nonzero
  test result.

## `test_velocity_gauge.cpp`

- `close()` compares complex transition moments within a strict tolerance.
- `velocity_index()` reproduces the direction-major storage index used for a
  minimal locally owned electron-hole pair.
- `main()` checks the TDA `vX/gap` contraction, the full-BSE
  `-conj(v)Y/gap` term, complex conjugation for imaginary velocity elements,
  rejection of a zero Kohn-Sham gap, the directional/isotropic oscillator
  prefactor including `1/Nk`, and the peak of the normalized Lorentzian
  spectrum.

## `test_profiler.cpp`

- `require()` reports failed profiler invariants without another test-framework
  dependency.
- `exercise_exception_path()` checks that `ScopedTimer` stops its timer while
  unwinding an exception.
- `main()` checks nested entries, accumulated call counts, CPU/wall-time
  accessors, exception-safe scope cleanup, and the timing-table headings and
  indentation.

## `test_spectrum_mpi.cpp`

- `velocity_index()` reproduces the compact, direction-major local-pair
  `velocity_mo` storage layout.
- `main()` places the only nonzero electron-hole contribution in rank 1's
  local pair block, checks that each rank stores only its own velocity/gap
  data, and verifies that the reduced transition dipole reaches rank 0. It
  also checks independent rank-specific amplitude file write/read.

## `test_memory.cpp`

- Checks capacity accounting, shared RI storage deduplication, move/release
  behavior, peak/allocation provenance and nonzero live residuals.
- Runs on 1, 2 and 4 MPI ranks to verify minima, maxima, averages and root-only
  output, plus local scopes, subcommunicators and rank-local exception unwinding.
- `ScopedTimer` is safe before MPI initialization and after finalization; the
  serial profiler test also exercises operation without MPI initialization.

## `test_bse_files.cpp`

- `test_coarse_qp_reader()` checks that `bse_use_fine_kgrid = 0` reads
  `energy_qp`, converts Hartree energies to Rydberg, skips core states, and
  computes direct and indirect gaps on the coarse grid.
- `test_uniform_qp_lookup()` checks indexed assignment on shifted/permuted
  meshes with periodic-equivalent and rounded coordinates, for both QP formats.
  It rejects duplicates, missing/off-grid data, duplicate grid sites and a
  10000-point mesh whose rounded QP coordinates are genuinely ambiguous.
- `test_aims_reader_view()` checks producer detection, Coulomb-name and
  pre-converted velocity symlinks, and the non-destructive reader view. A
  separate case verifies that missing velocity input is rejected.
- `write_wc()` creates one minimal MatrixMarket Wc block for a selected atom
  pair. The test supplies a complete bare-Coulomb map but creates only the Wc
  file required by the local LibRI `list_I x list_J` pair. It checks that
  nonlocal files are not accessed and that the local interaction is `W = V + Wc`.

## `test_aims_mommat_to_velocity.py`

- Uses a nonsymmetric `(kz,ky,kx) = (2,1,3)` fixture to check the Fortran HDF5
  grid-axis correction, packed upper-triangle mapping, Hermitian
  reconstruction, unit conversion, and binary-v1 output layout.
- Checks that native per-k momentum streams produce byte-identical canonical
  output, and rejects missing/truncated blocks, invalid headers and nonfinite
  payloads without changing an existing output file.

## `test_generate_band.py`

- Checks complete, unshifted cubic meshes for grid sizes 2, 3, 4, 8, 10 and 14,
  with every point appearing exactly once in canonical order.
- Checks the command-line grid size, the backward-compatible default of 3,
  and rejection of grids smaller than 2. Sampling instructions do not generate
  quasiparticle energies.

## `test_hchain_local.py`

- Checks k-grid selection, separate SCF/export preparation, relative paths in
  the full baseline and refusal to overwrite an existing calculation.
- Rejects incomplete or ungapped SCF results, incomplete/nonpositive BSE
  energies and nonfinite optical output.

## `test_hchain_double_grid.py`

- Checks FHI-aims legacy `(AO, band)` text to packed `(band, AO)` conversion
  with a rectangular fixture, dense-grid occupation weights and source preservation.
- Rejects incompatible bases, incomplete KS exports, existing work directories,
  nonfinite spectra and numerical discrepancies; failed rechecks remove stale pass reports.

## `test_hchain_chi0_double_grid.py`

- Checks exact complex response interpolation on odd/even coarse grids and
  noncommensurate fine grids, including symmetric Nyquist images.
- Checks the analytic diagonal dielectric result and non-positive rejection,
  complete legacy Coulomb payloads and q/-q reconstruction, MatrixMarket
  round trips, malformed/duplicate data and refusal to overwrite a calculation.

## `test_haydock.cpp`

- Compares the continued fraction against the exact spectral resolvent of a
  complex Hermitian matrix, including a degenerate eigenvalue, zero optical
  seed, iteration cutoff, non-Hermitian output and nonfinite input.

## `test_elpa_solver.cpp`

- `require()` and the two `require_close()` overloads provide Boolean, real,
  and complex assertions without adding another test-framework dependency.
- `make_descriptor()` creates the block-cyclic descriptor used by the serial
  and two-rank ELPA tests.
- `localize()` maps a deterministic dense matrix onto the local block-cyclic
  storage owned by each MPI rank.
- `globalize()` is a test-only MPI reduction that constructs dense references;
  production spectrum code never globalizes amplitudes.
- `test_skew_solver_reference()` ports the fixed 2x2 A/B case from
  `hamilt_bse_solver_test.cpp` and checks the two positive full-BSE energies.
- `test_distributed_matrix_checks()` checks the ScaLAPACK/MPI Hermitian test
  for A and symmetric test for B with both valid and deliberately broken
  distributed matrices.
- `test_tda_solver_residual()` solves a
  deterministic Hermitian matrix with ELPA and checks `A v = omega v` for
  every eigenpair, then checks redistribution into the spectrum pair layout.
- `test_full_solver_residual_and_metric()` checks the residual of
  `[[A,B],[-conj(B),-conj(A)]]`, plus the full-BSE symplectic normalization
  `X^H X - Y^H Y = I` and redistribution of both X/Y components.
- `deterministic_hermitian()`, `deterministic_symmetric()`, and
  `full_bse_matrix()` construct reproducible test matrices without random
  seeds.
- `main()` initializes MPI, LibRPA/ELPA, and BLACS in the required order, runs
  the three solver tests in both one- and two-rank CTest entries, combines
  rank-local status, and finalizes cleanly.

## `test_molecular_lri_comm.cpp`

- `block_value()` defines deterministic complex LibRI k-block values.
- `main()` distributes source k blocks between two ranks, calls the migrated
  `transform_k_2dlocal()` over two communication batches, and checks every
  locally owned 2D block-cyclic matrix element against the dense formula.

## `test_uniform_qmesh.cpp`

- Checks uniform, shifted and rectangular meshes, unique transfer q keys,
  reciprocal equivalence, and fallback for irregular or duplicate meshes.
- Checks even-grid boundary phases against an explicit average of equivalent
  images, unchanged coarse-grid values, and unchanged odd-grid phases.
- Checks the AO/R image expansion against that same q-space correction on
  faces, edges and three-axis corners, with averaging enabled and disabled.

## `test_aims_screened.cpp`

- Checks a known Fourier polynomial on a finer mesh, time-reversal restoration,
  frequency selection and rejection of incomplete text matrix blocks.
- Compares the complete static binary W cache with the text reader in real
  space and rejects a missing binary q point, on one and two MPI ranks.

## Producer-data integration checks

- `test_fft_convolution` checks complex multi-field convolution against direct
  sums on 1×1×1, 1×1×4, 2×2×3 and 3×4×5 grids, transform normalization,
  shifted/permuted coordinates, wrapped R and even-corner image averaging.
  Invalid meshes/buffers are rejected; without FFTW it checks the unavailable
  backend error instead of silently skipping.
- `test_tr_virtual` checks complex-unitary vertex fits and complete-complex W
  transfer, ill-conditioned/singular/out-of-subspace and nonfinite inputs,
  shifted/permuted and ambiguous k coordinates, local partner slots, nonempty
  orbit-owner fallback and balanced Si14 ownership. Geometry is initialized once;
  fit and cache payloads remain local to each owner.
- `test_kmpi_contractions` checks both W G and W conj(G) against complex
  GEMM for rectangular auxiliary spaces, real/tiny-imaginary/fully complex W,
  and cache-budget boundaries (including optional U payload), overflow rejection
  and representative grid sizes.
- `test_ri_streaming` compares all four A/B interaction contributions across
  batch sizes, then compares independent k-pair and AO/R TDA actions against
  the explicit distributed matrix on three complex probes for singlet, RPA
  and triplet coefficients. Run it from
  a complete input directory; it supports aims W/chi0 and LibRPA Wc inputs.
  With FFTW, it also checks the FFT action on the same probes and weights.
  `--benchmark` skips dense matrices, first checks available actions agree, then
  times them separately after warmup using three batches and adaptive repetitions.
  For large grids, `--benchmark-apply` times only `bse_haydock_apply` and reports
  `reference_checked=0`; validate its spectrum separately against ELPA. It checks
  finite output without constructing or timing the k-pair reference.
  Each batch may use one repetition when an individual action exceeds 0.1 seconds.
  With `realspace_kmpi`, dense/streamed/FFT references use a separate Cartesian
  object. The test also checks that a distributed-k object rejects all six
  incompatible reference methods (five without FFTW); legacy objects retain
  direct FFT coverage.
  `bse_kmpi_cache_mb 0` forces the bounded contractions; a small positive
  budget exercises size-based fallback. `LIBBSE_TEST_MIXED_CACHE_BUDGET=1`
  gives alternating k owners zero budget to check collective agreement.
  `LIBBSE_TEST_COMPLEX_W=1` rotates both candidate and explicit-reference W by
  0.8+0.3i; it retains all Hx comparisons and skips only the Hermiticity check
  invalidated by that synthetic rotation. `--benchmark-apply` also saves the
  complete complex result in `hx.bin` for separate elementwise comparison.
- `LibBSE_real_Si222_haydock_native_w` compares the matrix-free 2v2c Si
  Haydock spectrum with the bundled frozen ELPA spectrum, including energy-grid
  and finiteness checks. It runs under the `real_si` label.
- With FFTW enabled, `LibBSE_real_Si222_haydock_fft_native_w` and its `_mpi2`
  variant check the FFT spectrum against the same frozen ELPA reference.
- `LibBSE_real_Si222_haydock_gagq_native_w` and its `_mpi2` variant compare
  GAGQ x spectra with the same frozen ELPA reference.
- `LibBSE_real_Si222_haydock_atom_partition` uses four MPI ranks and the same
  frozen spectrum to check the R backend's atom-pair partition.
- `LibBSE_real_Si222_haydock_chi0_headwing_mpi2` checks the atom partition with
  distributed RI input and analytic head/wing screening against frozen ELPA.
- `LibBSE_real_Si222_haydock_kmpi_native_w_mpi8` and
  `LibBSE_real_Si222_haydock_kmpi_chi0_headwing_mpi8` check the distributed-k
  backend with eight MPI ranks, x/GAGQ spectra and unchanged frozen ELPA
  references, including analytic head/wing screening from distributed RI data.


### Distributed Haydock recurrence

`LibBSE_haydock_mpi1`, `LibBSE_haydock_mpi2` and `LibBSE_haydock_mpi4` compare complex Hermitian
recurrences and resolvents against the replicated implementation. They cover
empty local slices, zero optical seeds, subgroup communicators, collective
rejection of malformed/nonfinite input and inconsistent iteration limits, and
partitioned history payloads. The existing `LibBSE_haydock` test also exercises
the serial API without MPI initialization. Euler additionally runs 6 and 16 ranks
and compares small native/shifted H-chain spectra against independent ELPA
results for all three LRI actions. Late coefficients in nearly exhausted
subspaces are sensitive to MPI summation order; the physical gates use the
spectrum, the seed norm, low-order moments and early recurrence coefficients.

### Automated LRI kernel and CI-report gates

`LibBSE_lri_kernel_*` registers eight real-input cases under `lri_kernel`.
They compare independent explicit matrices with streamed, AO/R and optional FFT
actions, require each rank's expected cached/bounded route, and require actual
TR reuse where applicable. See the [case coverage](../physical_testcases/lri_kernels/README.md).
These correctness checks run for both FFTW configurations without timing thresholds.

`LibBSE_ci_report` checks discovery from the configured CTest inventory, missing
new tests, missing inventory/JUnit, duplicate tests, skips and failures.
`LibBSE_lri_kernel_report` checks that missing numerical comparisons, nonfinite
or excessive errors, wrong cache routes and absent TR reuse cannot pass the gate.

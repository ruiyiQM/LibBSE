# FHI-aims -> LibRPA -> LibBSE: H-chain development workflows

Author: Ruiyi Zhou.

These small workflows compare ELPA and Haydock with the same electronic data
and LRI kernel. They are software checks, not converged predictions for an
isolated one-dimensional material. They do not include FHI-aims source or an
executable.

## Model and prerequisites

- Two H atoms with alternating 1.0 and 1.5 Angstrom bonds along z; the cell is
  12 x 12 x 2.5 Angstrom and uses three-dimensional periodic boundary conditions.
- Nonmagnetic PBE with the first-tier H settings in `control.in`. The reference
  export has 10 AO and 34 auxiliary functions. Basis, vacuum, frequency and
  k-grid convergence must be assessed separately.
- Start with `1x1x4`, then `1x1x8`; the BSE window is one occupied and four
  virtual bands, giving `4*Nk` transitions. The dimerized chain is required to
  have a finite gap before proceeding.
- Use a FHI-aims build supporting `output librpa mommat` and the interface
  files required by LibRPA. Native `mommat_ks_kpt_*.dat` exports are supported;
  Fortran HDF5 is not required for this route. The script uses `KS_method serial`.
- Build LibBSE as described in the [main README](../../README.md), adding
  `-DLIBBSE_BUILD_LIBRPA_DRIVER=ON` to produce `librpa/chi0_main.exe`.
  Install the [Python requirements](../../regression_test/backend/requirements.txt).
  The optional FFT action requires `-DLIBBSE_ENABLE_FFTW=ON`.

The canonical velocity converter uses anti-Hermiticity to reconcile native
lower-triangle gradients with the upper-triangle HDF5 convention. Regenerate
native-derived `velocity_matrix` files produced before that correction.

## Single-grid baseline

From the repository root, use a new working directory and supply executable
paths for your installation:

```bash
ulimit -s hard
python3 examples/09_hchain_local/run.py \
  --stage baseline --nk 4 --work build/hchain-k4 \
  --aims-exe /path/to/aims.x \
  --librpa-exe /path/to/build/librpa/chi0_main.exe \
  --libbse-exe /path/to/build/LibBSE
```

Repeat with `--nk 8 --work build/hchain-k8` for the fine-grid baseline.
`run.py` executes FHI-aims, LibRPA and LibBSE as separate single-MPI processes
with one OpenMP/BLAS thread. Each executable uses its own linked MPI runtime.
The baseline computes actual G0W0 QP energies and Wc, then an ELPA singlet-TDA
spectrum; it does not apply a scissor or model screening. The lowest imaginary
frequency is used as a static-limit approximation, not relabeled as zero.

```text
working-directory/
  electronic/   FHI-aims data, logs and canonical velocity_matrix
  gw/           LibRPA data, energy_qp and librpa.d/Wc_*
  bse/          LibBSE data, excitation states and optical spectra
  summary.json  Completed stages and numerical checks
```

Use `--stage scf` or `--stage export` for individual producer stages.
`--prepare-only` writes inputs without requiring the executables. Existing
working directories are never overwritten. Each executable stage has a
600-second timeout; failures retain logs. The runner requires completed,
gapped SCF output, complete exports, positive finite BSE energies and finite
optical data before writing a successful summary.

## Fixed-screening double grid

After both baseline calculations finish:

```bash
python3 examples/09_hchain_local/double_grid.py \
  --coarse build/hchain-k4 --fine build/hchain-k8 \
  --work build/hchain-4to8 --executable /path/to/build/LibBSE --run
```

This workflow shares coarse RI/Coulomb/Wc, dense KS/QP data and the same
AO/R-interpolated optical operator between ELPA and Haydock. Dense QP energies
come from the completed fine GW calculation; the script does not introduce a
new QP interpolation. It enables `bse_w_boundary_average true` for equivalent
boundary images on even coarse grids.

The script converts full-grid FHI legacy text orbitals from `(AO, band)` to
packed `(band, AO)` input. It requires the same geometry, AO basis and band
count and complete, unshifted grids in canonical order. Fine/coarse grid ratios
need not be integers. Source calculations are read without modification.

The new directory contains `electronic/`, `elpa/`, `haydock/`, `inputs.json`
and, after successful comparison, `comparison.json`. Omit `--run` to prepare
inputs only. Larger grids may require increasing `bse_haydock_iterations` in
the prepared input; the default cap is not a convergence guarantee.

The comparison checks all directional and isotropic spectra on identical
energy grids using a combined absolute/relative criterion. A failed recheck
removes a stale success report. Agreement means that the two solvers evaluated
the same interpolated kernel; it does not establish coarse-screening or
physical k-grid convergence. Haydock writes spectra and recurrence coefficients,
not per-state excitation energies or eigenvectors.

## Experimental bare-response interpolation control

```bash
python3 examples/09_hchain_local/chi0_double_grid.py \
  --coarse build/hchain-k4 --fine build/hchain-k8 \
  --work build/hchain-chi0-4to8 \
  --librpa-exe /path/to/build/librpa/chi0_main.exe \
  --libbse-exe /path/to/build/LibBSE --apply fft
```

This control exports coarse bare `chi0(R)`, interpolates it to the target q
points and screens with the target full/cut Coulomb matrices. Gamma is treated
separately with the coarse response and target Gamma Coulomb, retaining the
baseline scalar-head convention. Fine GW supplies QP energies and the reference
spectrum; its W is not used to construct the candidate screening.

The producer must support `output_chi0_rf` and the `LIBRPA_CHI0_RF 1` format
used by the pinned LibRPA dependency. This script is restricted to the supplied
two-H model, 10 AO, 34 auxiliary functions, legacy single-rank exports and
`1x1xNk` grids. It requires completed coarse/fine baselines, including the fine
ELPA reference. It preserves inputs and rejects existing work directories,
incomplete matrices, incompatible metadata and nonpositive dielectric matrices.

Outputs include the coarse response, target Wc(R), ELPA/Haydock spectra and a
comparison report. `pass` checks same-kernel solver agreement;
`physical_relative_L2_xyzi` separately reports disagreement with the fully
fine-grid reference. These checks do not establish physical convergence or
validate a universal Gamma interpolation prescription.

This route retains the complete fine W(R) support. Its default `--apply
realspace` action therefore costs O(Nfine^2) for fixed cell bases; `--apply fft`
uses the same support with O(Nfine log Nfine) action cost. No real-space tails
or auxiliary ranks are discarded. Response interpolation costs O(Ncoarse*Nfine),
and dielectric inversion costs O(Nfine*Naux^3), separately from the eigensolver.

## Solver costs and validation

For fixed cell, AO, auxiliary and band sizes, the standard fixed-screening
AO/R action costs O(Nk*NR), where NR is the retained coarse real-space support.
It is linear in fine Nk only while NR stays fixed. The optional FFT action
costs O(Nk log Nk), with a substantial auxiliary-pair prefactor. Neither result
claims linear scaling with atom count or includes SCF/GW construction.

With D = Nocc*Nvirt*Nk and m actual Hx calls, complete reorthogonalization adds
O(m^2*D) work and O(m*D) history globally. Distributed Haydock partitions that
history over MPI ranks, while the two global Hx scratch vectors remain
replicated. Convergence must be checked for the chosen energy window and
broadening; more quadrature nodes do not represent additional Hx calls.

```bash
ctest --test-dir build -R 'LibBSE_hchain_' --output-on-failure
ctest --test-dir build -LE 'integration|real_si|lri_kernel' --output-on-failure
ctest --test-dir build -L 'real_si|lri_kernel' --output-on-failure
```

The three H-chain preparation tests need no producer executable. They check
input preservation, packing/layout, malformed exports, screening interpolation
and rejection of invalid comparison results. The separate bundled H-chain/Si
[LRI kernel regressions](../../regression_test/physical_testcases/lri_kernels/README.md)
exercise actual matrix actions, including distributed-k caching and TR reuse.
They do not regenerate FHI-aims or GW data in CI.

For a prepared complete dataset, `test_ri_streaming --benchmark` compares
available actions after a correctness check and warmup. Its Hx timings exclude
file reading and initialization; they do not measure complete solver speedups.
Numerical convergence, input provenance and benchmark records should be kept
with the corresponding calculation rather than treated as portable performance
promises for this example.

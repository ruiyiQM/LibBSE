#!/usr/bin/env python3
"""Prepare and optionally compare ELPA/Haydock with coarse screening and dense KS/QP data."""
# Author: Ruiyi Zhou.

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent


def read_export(directory):
    words = (directory / "band_out").read_text().split()
    nk, spins, bands, nao = map(int, words[:4])
    if min(nk, bands, nao) <= 0 or spins != 1:
        raise ValueError("double-grid example requires a nonmagnetic scalar export")
    cursor = 5
    eigenvalues = []
    for k in range(nk):
        if list(map(int, words[cursor:cursor + 2])) != [k + 1, 1]:
            raise ValueError("invalid band_out k/spin order")
        cursor += 2
        block = np.asarray(words[cursor:cursor + 4 * bands], float).reshape(bands, 4)
        if not np.array_equal(block[:, 0], np.arange(1, bands + 1)) or not np.isfinite(block).all():
            raise ValueError("invalid band_out states")
        eigenvalues.append(block)
        cursor += 4 * bands
    if cursor != len(words):
        raise ValueError("unexpected band_out payload")
    lines = (directory / "bz_sampling_out").read_text().splitlines()
    grid = tuple(map(int, lines[0].split()))
    if len(grid) != 3 or min(grid) <= 0 or np.prod(grid) != nk:
        raise ValueError("inconsistent full k grid")
    kpoints = np.asarray([line.split()[2:5] for line in lines[2:2 + nk]], float)
    expected = np.asarray(list(np.ndindex(grid)), float) / np.asarray(grid)
    if kpoints.shape != (nk, 3) or not np.allclose(kpoints, expected, atol=1.e-10, rtol=0):
        raise ValueError("example requires the complete unshifted grid in canonical order")
    wavefunctions = {}
    for source in sorted(directory.glob("KS_eigenvector_*.txt")):
        tokens = source.read_text().split()
        cursor = 0
        while cursor < len(tokens):
            k = int(tokens[cursor]) - 1
            cursor += 1
            if not 0 <= k < nk or k in wavefunctions:
                raise ValueError("invalid or repeated KS k-point block")
            values = np.asarray(tokens[cursor:cursor + 2 * bands * nao], float).reshape(nao, bands, 2)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite KS coefficients")
            # FHI legacy text is (AO, band); packed band reader wants (band, AO).
            wavefunctions[k] = np.ascontiguousarray((values[:, :, 0] + 1j * values[:, :, 1]).T, dtype="<c16")
            cursor += 2 * bands * nao
    if len(wavefunctions) != nk:
        raise ValueError("incomplete KS wavefunctions")
    return nk, bands, nao, grid, kpoints, eigenvalues, wavefunctions


def prepare(coarse, fine, work):
    if work.exists():
        raise FileExistsError("use a new work directory; existing calculations are never overwritten")
    coarse, fine = coarse.resolve(), fine.resolve()
    for name in ["geometry.in", "basis_out"]:
        if (coarse / "electronic" / name).read_bytes() != (fine / "electronic" / name).read_bytes():
            raise ValueError(f"coarse and dense exports have different {name}")
    c = read_export(coarse / "electronic")
    f = read_export(fine / "electronic")
    if c[1] < 5 or c[1:3] != f[1:3] or any(b < a for a, b in zip(c[3], f[3])):
        raise ValueError("dense grid must be at least as fine and use the same bands and AO basis")
    if (not (coarse / "electronic/velocity_matrix").is_file()
            or not (fine / "gw/energy_qp").is_file()
            or not any((coarse / "gw/librpa.d").glob("Wc_Mu_*_ifreq_0.mtx"))):
        raise ValueError("missing coarse velocity, dense QP energies or coarse Wc")
    electronic = work / "electronic"
    electronic.mkdir(parents=True)
    for source in (coarse / "electronic").iterdir():
        if source.is_file():
            (electronic / source.name).symlink_to(source)
    with (electronic / "band_kpath_info").open("x") as output:
        output.write(f"{f[2]} {f[1]} 1 {f[0]}\n")
        np.savetxt(output, f[4], fmt="%.17e")
    for k in range(f[0]):
        with (electronic / f"band_KS_eigenvector_k_{k + 1:05d}.txt").open("xb") as output:
            output.write(f[6][k].tobytes())
        with (electronic / f"band_KS_eigenvalue_k_{k + 1:05d}.txt").open("x") as output:
            for state, occupation, energy_ha, energy_ev in f[5][k]:
                output.write(f"{int(state)} 1 {occupation / f[0]:.17e} {energy_ha:.17e} {energy_ev:.17e}\n")
    text = (HERE / "libbse.in").read_text()
    text = text.replace("qp_data ../gw/energy_qp", f"qp_data {fine / 'gw/energy_qp'}")
    text = text.replace("screened_dir ../gw/librpa.d", f"screened_dir {coarse / 'gw/librpa.d'}")
    text = text.replace("bse_use_fine_kgrid 0", "bse_use_fine_kgrid 1")
    text += "\nbse_w_boundary_average true\n"
    for solver in ["elpa", "haydock"]:
        directory = work / solver
        directory.mkdir()
        settings = text
        if solver == "haydock":
            settings = settings.replace("bse_solver elpa", "bse_solver haydock").replace(
                "out_bse_eigenvectors true", "out_bse_eigenvectors false")
        (directory / "libbse.in").write_text(settings)
    metadata = {"coarse_nk": c[0], "dense_nk": f[0], "tda_dimension": 4 * f[0],
                "coarse_source": str(coarse), "dense_source": str(fine),
                "screening_source": str(coarse / "gw/librpa.d"),
                "qp_source": str(fine / "gw/energy_qp")}
    (work / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def compare(work):
    # A failed recheck must not leave an earlier successful report behind.
    (work / "comparison.json").unlink(missing_ok=True)
    tables = [np.loadtxt(work / solver / "libbse.d/spectrum_singlet_tda.dat") for solver in ["elpa", "haydock"]]
    a, b = tables
    if a.ndim != 2 or a.shape != b.shape or a.shape[1] != 5 or not all(np.isfinite(t).all() for t in tables):
        raise ValueError("invalid ELPA/Haydock spectra")
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("spectrum energy grids differ")
    errors = []
    for axis in range(1, 5):
        difference = np.linalg.norm(a[:, axis] - b[:, axis])
        scale = np.linalg.norm(a[:, axis])
        if difference > 1.e-10 + 1.e-8 * scale:
            raise ValueError(f"ELPA/Haydock spectrum differs in column {axis}")
        errors.append(float(difference / max(scale, 1.e-30)))
    result = {"pass": True, "relative_spectrum_errors_xyzi": errors}
    (work / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", required=True, type=Path, help="completed run.py baseline directory")
    parser.add_argument("--fine", required=True, type=Path, help="completed dense run.py baseline directory")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--executable", type=Path, default=HERE.parents[1] / "build-local/LibBSE")
    args = parser.parse_args()
    work = args.work.resolve()
    metadata = prepare(args.coarse, args.fine, work)
    if args.run:
        for solver in ["elpa", "haydock"]:
            with (work / solver / "libbse.out").open("w") as output, (work / solver / "libbse.err").open("w") as error:
                subprocess.run([str(args.executable.resolve())], cwd=work / solver,
                               env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                               stdout=output, stderr=error, timeout=600, check=True)
        metadata.update(compare(work))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

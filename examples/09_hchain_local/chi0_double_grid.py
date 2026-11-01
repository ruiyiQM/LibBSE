#!/usr/bin/env python3
"""H-chain screening control: interpolate bare chi0, then screen on the fine grid.

Uses coarse GW and the fine KS export, including its bare Coulomb matrices.
Fine GW contributes QP energies and the optional reference spectrum, never W.
This experimental route retains the full fine W(R) support.
"""
# Author: Ruiyi Zhou.
import argparse
import json
import os
from pathlib import Path
import re
import struct

import numpy as np

from double_grid import read_export
from run import run_stage

HERE = Path(__file__).resolve().parent
NAUX = 34


def read_coulomb(path, nk):
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"truncated legacy Coulomb file: {path}")
    total, local = struct.unpack_from("<2i", raw)
    if not 1 <= local <= total <= nk:
        raise ValueError("requires a single-rank FHI legacy Coulomb export")
    result = np.zeros((nk, NAUX, NAUX), complex)
    regions = {}
    offset = 8
    for _ in range(local):
        if len(raw) - offset < 32:
            raise ValueError("truncated Coulomb block header")
        shape = struct.unpack_from("<5i", raw, offset)
        ik, weight = struct.unpack_from("<id", raw, offset + 20)
        offset += 32
        end = offset + 16 * NAUX * NAUX
        if (shape != (NAUX, 1, NAUX, 1, NAUX) or not 1 <= ik <= nk
                or ik - 1 in regions or not np.isfinite(weight) or weight <= 0 or end > len(raw)):
            raise ValueError("invalid, duplicate or incomplete Coulomb block")
        block = np.frombuffer(raw, dtype="<c16", count=NAUX * NAUX, offset=offset).reshape(NAUX, NAUX)
        if not np.isfinite(block).all() or np.max(np.abs(block - block.conj().T)) > 1e-9:
            raise ValueError("nonfinite or non-Hermitian Coulomb block")
        result[ik - 1] = block
        regions[ik - 1] = slice(offset, end)
        offset = end
    if offset != len(raw) or 0 not in regions:
        raise ValueError("missing Gamma or trailing Coulomb payload")
    for ik in range(nk):
        if ik not in regions:
            if (-ik) % nk not in regions:
                raise ValueError("incomplete q/-q Coulomb grid")
            result[ik] = result[(-ik) % nk].conj()
    return result, raw, regions[0]


def read_real_space(directory, prefix, nk):
    result = np.zeros((nk, NAUX, NAUX), complex)
    seen = set()
    for file in directory.glob(f"{prefix}_Mu_*_ifreq_0.mtx"):
        match = re.fullmatch(rf"{prefix}_Mu_(\d+)_Nu_(\d+)_iR_(\d+)_ifreq_0\.mtx", file.name)
        if match is None:
            raise ValueError("unexpected real-space matrix filename")
        i, j, ir = map(int, match.groups())
        lines = file.read_text().splitlines()
        if not lines or lines[0] != "%%MatrixMarket matrix coordinate complex general":
            raise ValueError("invalid MatrixMarket header")
        cells = [re.search(r"\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)", line)
                 for line in lines if line.startswith("%")]
        cells = [tuple(map(int, cell.groups())) for cell in cells if cell is not None]
        if (i not in (0, 1) or j not in (0, 1) or not 0 <= ir < nk
                or (i, j, ir) in seen or cells != [(0, 0, ir - nk // 2)]):
            raise ValueError("invalid atom pair, R cell or duplicate block")
        seen.add((i, j, ir))
        entries = [line for line in lines if line and not line.startswith("%")]
        if not entries:
            raise ValueError("missing matrix dimensions")
        nr, nc, nz = map(int, entries[0].split())
        if (nr, nc) != (17, 17) or not 0 <= nz <= 289 or len(entries) != nz + 1:
            raise ValueError("incomplete or incompatible auxiliary matrix")
        indices = set()
        for line in entries[1:]:
            r, c, x, y = line.split()
            r, c, value = int(r) - 1, int(c) - 1, complex(float(x), float(y))
            if not 0 <= r < 17 or not 0 <= c < 17 or (r, c) in indices or not np.isfinite(value):
                raise ValueError("invalid, repeated or nonfinite matrix entry")
            indices.add((r, c))
            result[ir, i * 17 + r, j * 17 + c] = value
    if len(seen) != 4 * nk:
        raise ValueError("incomplete real-space response grid")
    return result


def interpolate_response(chir, nq):
    if chir.ndim != 3 or chir.shape[0] < 1 or chir.shape[1] != chir.shape[2] or nq < 1:
        raise ValueError("invalid response shape or target grid")
    nr = len(chir)
    cells = np.arange(nr) - nr // 2
    phase = np.exp(2j * np.pi * np.arange(nq)[:, None] / nq * cells[None, :])
    if nr % 2 == 0:
        phase[:, 0] = phase[:, 0].real
    response = np.einsum("qr,rij->qij", phase, chir)
    if not np.isfinite(response).all() or np.max(np.abs(response - response.swapaxes(1, 2).conj())) > 1e-10:
        raise ValueError("interpolated response is nonfinite or non-Hermitian")
    return response


def sqrt_positive(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.conj().T) / 2)
    return (vectors * np.sqrt(np.where(values > 1e-8, values, 0))) @ vectors.conj().T


def screen_response(chiq, full, cut, gamma_wc):
    wc = np.zeros_like(chiq)
    wc[0] = gamma_wc
    for ik in range(1, len(chiq)):
        a, b = sqrt_positive(full[ik]), sqrt_positive(cut[ik])
        epsilon = np.eye(chiq.shape[1]) - a @ chiq[ik] @ a
        if np.linalg.eigvalsh((epsilon + epsilon.conj().T) / 2).min() <= 0:
            raise ValueError("non-positive dielectric matrix")
        wc[ik] = b @ (np.linalg.inv(epsilon) - np.eye(chiq.shape[1])) @ b
    if not np.isfinite(wc).all() or np.max(np.abs(wc - wc.swapaxes(1, 2).conj())) > 1e-10:
        raise ValueError("nonfinite or non-Hermitian screened interaction")
    return wc


def write_screening(directory, wcq):
    nk = len(wcq)
    cells = np.arange(nk) - nk // 2
    wcr = np.einsum("rq,qij->rij", np.exp(-2j * np.pi * cells[:, None] * np.arange(nk)[None, :] / nk) / nk, wcq)
    directory.mkdir()
    for ir, cell in enumerate(cells):
        for i in range(2):
            for j in range(2):
                with (directory / f"Wc_Mu_{i}_Nu_{j}_iR_{ir}_ifreq_0.mtx").open("x") as out:
                    out.write(f"%%MatrixMarket matrix coordinate complex general\n%\n% Wc at iR {ir} ( 0 0 {cell} ) and ifreq 0\n%\n17 17 289\n")
                    for a in range(17):
                        for b in range(17):
                            value = wcr[ir, i * 17 + a, j * 17 + b]
                            out.write(f"{a + 1} {b + 1} {value.real:.17e} {value.imag:.17e}\n")


def prepare(coarse, fine, work):
    if work.exists():
        raise FileExistsError("use a new work directory; previous calculations are preserved")
    coarse, fine = coarse.resolve(), fine.resolve()
    for name in ("geometry.in", "basis_out"):
        if (coarse / "electronic" / name).read_bytes() != (fine / "electronic" / name).read_bytes():
            raise ValueError(f"coarse and fine exports have different {name}")
    c, f = read_export(coarse / "electronic"), read_export(fine / "electronic")
    if c[1:3] != (10, 10) or f[1:3] != (10, 10) or c[3][:2] != (1, 1) or f[3][:2] != (1, 1) or f[0] < c[0]:
        raise ValueError("this control requires the 2-H, 10-AO chain and a finer 1x1xNk grid")
    if not (fine / "gw/energy_qp").is_file():
        raise ValueError("fine-grid QP data are required")
    for name in ("electronic/velocity_matrix", "bse/libbse.d/spectrum_singlet_tda.dat",
                 "bse/libbse.d/Excitation_Energy_singlet.dat"):
        if not (fine / name).is_file() or (fine / name).stat().st_size == 0:
            raise ValueError(f"incomplete fine baseline: {name}")
    settings = (coarse / "gw/librpa.in").read_text()
    if ("input_dir = ../electronic" not in settings or "output_dir = librpa.d" not in settings
            or "option_dielect_func = 0" not in settings or "sqrt_coulomb_threshold = 1e-8" not in settings):
        raise ValueError("requires the baseline scalar-head LibRPA settings")
    gamma_files = {}
    for name in ("coulomb_mat_0.txt", "coulomb_cut_0.txt"):
        _, raw, region = read_coulomb(coarse / "electronic" / name, c[0])
        _, target, target_region = read_coulomb(fine / "electronic" / name, f[0])
        changed = bytearray(raw)
        changed[region] = target[target_region]
        gamma_files[name] = changed
    electronic = work / "electronic"
    electronic.mkdir(parents=True)
    for source in (coarse / "electronic").iterdir():
        if source.is_file() and source.name not in gamma_files:
            (electronic / source.name).symlink_to(source.resolve())
    for name, raw in gamma_files.items():
        (electronic / name).write_bytes(raw)
    gw = work / "gw"
    gw.mkdir()
    (gw / "librpa.in").write_text(settings + "\noutput_chi0_rf = t\n")
    return c[0], f[0]


def execute(coarse, fine, work, librpa, libbse, apply="realspace"):
    nc, nf = prepare(coarse, fine, work)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run_stage(librpa, work / "gw", "librpa", env)
    response_dir = work / "gw/librpa.d/chi0_lowest"
    info = (response_dir / "chi0_rf.info").read_text().split()
    if len(info) != 7 or info[:2] != ["LIBRPA_CHI0_RF", "1"] or list(map(int, info[3:])) != [NAUX, 1, 1, nc] or not np.isfinite(float(info[2])):
        raise ValueError("incompatible response metadata")
    chir = read_real_space(response_dir, "Chi0", nc)
    gamma = read_real_space(work / "gw/librpa.d", "Wc", nc).sum(axis=0)
    full = read_coulomb(fine / "electronic/coulomb_mat_0.txt", nf)[0]
    cut = read_coulomb(fine / "electronic/coulomb_cut_0.txt", nf)[0]
    wcq = screen_response(interpolate_response(chir, nf), full, cut, gamma)
    write_screening(work / "screened", wcq)
    settings = (HERE / "libbse.in").read_text()
    settings = settings.replace("input_dir ../electronic", f"input_dir {fine / 'electronic'}")
    settings = settings.replace("qp_data ../gw/energy_qp", f"qp_data {fine / 'gw/energy_qp'}")
    settings = settings.replace("screened_dir ../gw/librpa.d", f"screened_dir {work / 'screened'}")
    for solver in ("elpa", "haydock"):
        run = work / solver
        run.mkdir()
        text = settings
        if solver == "haydock":
            text = text.replace("bse_solver elpa", "bse_solver haydock").replace("out_bse_eigenvectors true", "out_bse_eigenvectors false")
            text += f"\nbse_haydock_apply {apply}\n"
        (run / "libbse.in").write_text(text)
        run_stage(libbse, run, "libbse", env)
    reference = np.loadtxt(fine / "bse/libbse.d/spectrum_singlet_tda.dat")
    candidate = np.loadtxt(work / "haydock/libbse.d/spectrum_singlet_tda.dat")
    if reference.shape != candidate.shape or not np.array_equal(reference[:, 0], candidate[:, 0]) or not np.isfinite(reference).all():
        raise ValueError("invalid dense reference spectrum or mismatched energy grid")
    errors = np.linalg.norm(candidate[:, 1:] - reference[:, 1:], axis=0) / np.maximum(np.linalg.norm(reference[:, 1:], axis=0), 1e-30)
    energies = np.loadtxt(work / "elpa/libbse.d/Excitation_Energy_singlet.dat", ndmin=1)
    reference_energies = np.loadtxt(fine / "bse/libbse.d/Excitation_Energy_singlet.dat", ndmin=1)
    if energies.shape != reference_energies.shape or not np.isfinite(energies).all() or not np.isfinite(reference_energies).all():
        raise ValueError("invalid excitation-energy reference")
    from double_grid import compare
    report = compare(work)
    report.update(coarse_nk=nc, fine_nk=nf, response_frequency_ha=float(info[2]),
                  physical_relative_L2_xyzi=errors.tolist(),
                  pass_scope="same-kernel ELPA/Haydock; physical convergence is reported separately",
                  lowest_excitation_ev=float(energies.min() * 13.605693122994),
                  lowest_error_ev=float((energies.min() - reference_energies.min()) * 13.605693122994),
                  construction="coarse bare response, target bare Coulomb, coarse response with target Gamma V; fine GW W is not read",
                  retained_R_cells=nf,
                  haydock_apply=apply,
                  complexity=("response interpolation O(Nc*Nf), screening O(Nf*Naux^3), "
                              + ("FFT action O(Nf log Nf) for fixed AO/auxiliary/band basis"
                                 if apply == "fft" else "R action O(Nf^2) with full fine-R support")))
    (work / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--apply", choices=("realspace", "fft"), default="realspace")
    parser.add_argument("--librpa-exe", type=Path, default=HERE.parents[1] / "build-local/librpa/chi0_main.exe")
    parser.add_argument("--libbse-exe", type=Path, default=HERE.parents[1] / "build-local/LibBSE")
    args = parser.parse_args()
    print(json.dumps(execute(args.coarse.resolve(), args.fine.resolve(), args.work.resolve(),
                             args.librpa_exe.resolve(), args.libbse_exe.resolve(), args.apply), indent=2))


if __name__ == "__main__":
    main()

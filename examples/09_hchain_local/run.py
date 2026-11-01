#!/usr/bin/env python3
"""Run a single-rank H-chain SCF, export, or FHI-aims/LibRPA/LibBSE baseline."""
# Author: Ruiyi Zhou.

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPORT_SETTINGS = """\
qpe_calc gw_expt
frequency_points 16
anacon_type 0
default_prodbas_acc 1e-3
default_max_l_prodbas 3
periodic_gw_optimize_kgrid_symmetry inverse
output librpa mommat

"""


def prepare(work, nk, stage):
    if nk < 1:
        raise ValueError("nk must be positive")
    control = (HERE / "control.in").read_text()
    control = control.replace("k_grid 1 1 4\n", f"k_grid 1 1 {nk}\n")
    if stage == "export":
        control = EXPORT_SETTINGS + control
    work.mkdir(parents=True, exist_ok=False)
    (work / "control.in").write_text(control)
    shutil.copyfile(HERE / "geometry.in", work / "geometry.in")


def check_scf(output):
    if "Self-consistency cycle converged." not in output or "Have a nice day." not in output:
        raise ValueError("FHI-aims did not finish a converged SCF calculation")
    gaps = re.findall(r"overall HOMO-LUMO gap:\s*([-+0-9.eE]+)", output, re.IGNORECASE)
    if not gaps:
        raise ValueError("missing overall KS gap; inspect aims.out")
    gap = float(gaps[-1])
    if not math.isfinite(gap) or gap <= 1e-4:
        raise ValueError(f"H chain is not a gapped starting point: {gap} eV")
    return gap


def run_stage(executable, work, label, env):
    # Singleton MPI keeps each executable with its own linked MPI implementation.
    with (work / f"{label}.out").open("w") as out, (work / f"{label}.err").open("w") as err:
        subprocess.run([str(executable.resolve())], cwd=work, env=env,
                       stdout=out, stderr=err, check=True, timeout=600)


def check_bse(output_dir, nk):
    energies = [float(word) for word in
                (output_dir / "Excitation_Energy_singlet.dat").read_text().split()]
    if len(energies) != 4 * nk or any(not math.isfinite(e) or e <= 0 for e in energies):
        raise ValueError("BSE must return 4*nk finite, positive excitation energies")
    for name, columns in [("oscillator_strength_singlet_tda.dat", 6),
                          ("spectrum_singlet_tda.dat", 5)]:
        rows = [line.split() for line in (output_dir / name).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if not rows or any(len(row) != columns or any(not math.isfinite(float(x)) for x in row)
                           for row in rows):
            raise ValueError(f"invalid optical output: {name}")
        if name.startswith("oscillator") and len(rows) != len(energies):
            raise ValueError("oscillator strengths do not cover every BSE state")
    return min(energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="new output directory")
    parser.add_argument("--nk", type=int, default=4, help="k_grid = 1 1 NK")
    parser.add_argument("--stage", choices=["scf", "export", "baseline"], default="scf")
    parser.add_argument("--aims-exe", type=Path)
    parser.add_argument("--librpa-exe", type=Path,
                        default=HERE.parents[1] / "build-local/librpa/chi0_main.exe")
    parser.add_argument("--libbse-exe", type=Path, default=HERE.parents[1] / "build-local/LibBSE")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    executables = [args.aims_exe]
    if args.stage == "baseline":
        executables += [args.librpa_exe, args.libbse_exe]
    if not args.prepare_only:
        for executable in executables:
            if executable is None or not executable.is_file() or not os.access(executable, os.X_OK):
                parser.error(f"executable not found: {executable}; provide its path or use --prepare-only")
    stack_limit, _ = resource.getrlimit(resource.RLIMIT_STACK)
    if not args.prepare_only and stack_limit != resource.RLIM_INFINITY and stack_limit < 60 * 1024 * 1024:
        parser.error("run 'ulimit -s hard' in the shell before starting this script")
    work = args.work.resolve()
    electronic = work / "electronic" if args.stage == "baseline" else work
    if args.stage == "baseline":
        if work.exists():
            parser.error("use a new --work directory; existing calculations are never overwritten")
        prepare(electronic, args.nk, "export")
        for directory, name in [("gw", "librpa.in"), ("bse", "libbse.in")]:
            (work / directory).mkdir()
            shutil.copyfile(HERE / name, work / directory / name)
    else:
        prepare(work, args.nk, args.stage)
    if args.prepare_only:
        print(f"Prepared {args.stage} input: {work}")
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    run_stage(args.aims_exe, electronic, "aims", env)
    gap = check_scf((electronic / "aims.out").read_text())
    summary = {"stage": args.stage, "nk": args.nk, "ks_gap_ev": gap,
               "aims_executable": str(args.aims_exe.resolve()), "bse_run": False}
    if args.stage in ["export", "baseline"]:
        for name in ["basis_out", "band_out", "KS_eigenvector_0.txt", "Cs_data_0.txt",
                     "coulomb_cut_0.txt", "coulomb_mat_0.txt", "stru_out", "bz_sampling_out",
                     "vxc_out", "dielecfunc_out"]:
            if not (electronic / name).is_file() or (electronic / name).stat().st_size == 0:
                raise ValueError(f"missing interface output: {name}")
        for ik in range(1, args.nk + 1):
            if not (electronic / f"mommat_ks_kpt_{ik:06d}.dat").is_file():
                raise ValueError(f"missing native momentum block for k point {ik}")
        subprocess.run([sys.executable, str(HERE.parents[1] / "tools/aims_mommat_to_velocity.py"),
                        str(electronic)], check=True)
        summary["velocity_matrix"] = str(electronic / "velocity_matrix")
        nk, nspin, nbands, nao = map(int, (electronic / "band_out").read_text().split()[:4])
        if (nk, nspin) != (args.nk, 1) or nbands < 5:
            raise ValueError("export does not cover the requested grid and 1v4c window")
        summary.update(nbands=nbands, nao=nao, planned_tda_dimension=4 * nk)
    if args.stage == "baseline":
        run_stage(args.librpa_exe, work / "gw", "librpa", env)
        if not (work / "gw/energy_qp").is_file() or not any(
                (work / "gw/librpa.d").glob("Wc_Mu_*_ifreq_0.mtx")):
            raise ValueError("LibRPA did not produce QP energies and static-limit Wc")
        run_stage(args.libbse_exe, work / "bse", "libbse", env)
        gap_ry = check_bse(work / "bse/libbse.d", args.nk)
        summary.update(bse_run=True, lowest_excitation_ev=gap_ry * 13.605693122994)
    (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

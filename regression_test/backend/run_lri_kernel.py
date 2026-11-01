"""Compare real-input LRI actions with an independently assembled dense Hx."""
# Author: Ruiyi Zhou.

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import tomllib

from run_case import file_digest, parse_input
from run_real_si import extract_verified, options_for, restore_screening

FIXTURES = Path(__file__).resolve().parents[1] / 'physical_testcases/lri_kernels'


def validate_kernel(work, ranks, cached, require_tr, complex_w, fftw, owner_fallback=False):
    text = (work / 'run.out').read_text()
    maximum = 0.
    counts = {'RI_MATRIX_COMPARE': 12, 'RI_APPLY_COMPARE': 27,
              'RI_REAL_SPACE_COMPARE': 9, 'RI_REAL_SPACE_ADJOINT': 0 if complex_w else 6,
              'RI_FFT_COMPARE': 9 if fftw else 0}
    for marker, count in counts.items():
        errors = [float(x) for x in re.findall(r'^' + marker + r' .*max_abs=(\S+)', text, re.M)]
        if len(errors) != count or any(not math.isfinite(x) or x > 5e-12 for x in errors):
            raise ValueError(f'{marker}: missing comparisons or incorrect/nonfinite Hx')
        maximum = max([maximum, *errors])
    guards = 6 if fftw else 5
    if f'RI_KMPI_REFERENCE_GUARDS checked={guards}' not in text:
        raise ValueError('Distributed-k reference guards were not exercised')
    if complex_w and 'RI_SYNTHETIC_COMPLEX_W rotated=true hermitian_check=false' not in text:
        raise ValueError('Synthetic complex W was not exercised')
    accepted = 0
    for rank in range(ranks):
        log = (work / f'libbse.d/libri_rank_{rank}.log').read_text()
        if f'LRI_R_PARTITION mode=atom_k rank={rank} ' not in log:
            raise ValueError(f'Rank {rank} did not use distributed-k ownership')
        fields = [line for line in log.splitlines() if line.startswith('LRI_KMPI_FIELD ')]
        tr = [line for line in log.splitlines() if line.startswith('LRI_KMPI_TR_REUSE ')]
        if len(fields) != 6 or len(tr) != 6:
            raise ValueError(f'Rank {rank}: incomplete screened Hx records')
        if any(f'cached={int(cached)} ' not in line for line in fields):
            raise ValueError(f'Rank {rank}: expected cache/fallback route was not taken')
        if not cached and any('enabled=0 ' not in line for line in tr):
            raise ValueError('TR reuse enabled without its cache payload')
        if owner_fallback and (any('local_k=1 ' not in line for line in fields)
                               or any('enabled=0 ' not in line for line in tr)):
            raise ValueError('Single-k owners must retain the non-TR contraction')
        for line in tr:
            values = dict(re.findall(r'(\w+)=(\S+)', line))
            tested, reused = int(values['tested_pairs']), int(values['accepted_pairs'])
            residual = float(values['accepted_max'])
            if not 0 <= reused <= tested or not math.isfinite(residual) or residual > 1e-13:
                raise ValueError('Invalid TR acceptance record')
            accepted += reused
    if require_tr and accepted == 0:
        raise ValueError('No TR pair was reused; this would leave the optimized path untested')
    return dict(passed=True, max_error_Ry=maximum, accepted_TR_records=accepted,
                rank_Hx_records=6*ranks, cached=cached, fftw=fftw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    catalog = tomllib.loads((FIXTURES / 'cases.toml').read_text())
    parser.add_argument('--case', choices=sorted(catalog), required=True)
    parser.add_argument('--fftw', choices=['ON', 'OFF'], required=True)
    args = parser.parse_args()
    definition = catalog[args.case]
    work = (args.work / f'{args.case}_{time.time_ns()}').resolve()
    work.mkdir(parents=True)
    if definition['fixture'] == 'hchain':
        extract_verified('inputs.tar.gz', work / 'fixture', FIXTURES)
        source = work / 'fixture/hchain_coarse4_fine8'
        options = parse_input(source / 'libbse.in')
        for key in ('input_dir', 'qp_data', 'screened_dir'):
            options[key] = str(source / options[key])
    else:
        extract_verified('inputs.tar.xz', work / 'fixture')
        restore_screening(work / 'fixture')
        options = options_for(work / 'fixture', {'options': {'screened_format': 'fhi_aims_w'}})
    options.update(bse_solver='haydock', bse_haydock_apply='realspace_kmpi',
                   bse_nstates='-1', out_bse_eigenvectors='false',
                   bse_kmpi_cache_mb=str(definition['budget_mb']))
    (work / 'libbse.in').write_text(''.join(f'{k} {v}\n' for k, v in options.items()))
    env = dict(os.environ)
    for key in ('LIBBSE_TEST_COMPLEX_W', 'LIBBSE_TEST_MIXED_CACHE_BUDGET'):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS=str(definition['threads']),
               OMP_THREAD_LIMIT=str(definition.get('thread_limit', definition['threads'])),
               OPENBLAS_NUM_THREADS='1', OMPI_MCA_rmaps_base_oversubscribe='1')
    if definition.get('complex_w'):
        env['LIBBSE_TEST_COMPLEX_W'] = '1'
    if definition.get('mixed_budget'):
        env['LIBBSE_TEST_MIXED_CACHE_BUDGET'] = '1'
    exe = args.executable.resolve()
    command = ['mpirun', '-np', str(definition['ranks']), '--map-by', 'slot',
               '--bind-to', os.environ.get('LIBBSE_MPI_BIND_TO', 'none'), str(exe)]
    (work / 'execution.json').write_text(json.dumps(dict(
        command=command, executable_sha256=file_digest(exe), case=definition), indent=2)+'\n')
    with (work / 'run.out').open('w') as out, (work / 'run.err').open('w') as err:
        result = subprocess.run(command, cwd=work, env=env, stdout=out, stderr=err, timeout=1100)
    if result.returncode:
        raise RuntimeError(f'Kernel regression failed in {work}:\n'
                           + (work / 'run.err').read_text(errors='replace')[-6000:])
    report = validate_kernel(work, definition['ranks'], definition['cached'],
                             definition.get('require_tr', False), definition.get('complex_w', False),
                             args.fftw == 'ON', definition.get('owner_fallback', False))
    report.update(case=args.case, work=str(work))
    (work / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

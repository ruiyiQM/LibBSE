"""A missing numerical comparison or optimized route must fail the CI gate."""
# Author: Ruiyi Zhou.

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from run_lri_kernel import validate_kernel


class KernelReportTests(unittest.TestCase):
    def test_rejects_incomplete_nonfinite_or_unexercised_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'libbse.d').mkdir()
            text = ''.join((f'{name} spin=singlet max_abs=1e-15\n') * count for name, count in [
                ('RI_MATRIX_COMPARE', 12), ('RI_APPLY_COMPARE', 27),
                ('RI_REAL_SPACE_COMPARE', 9), ('RI_REAL_SPACE_ADJOINT', 6), ('RI_FFT_COMPARE', 9)])
            text += 'RI_KMPI_REFERENCE_GUARDS checked=6\n'
            output = work / 'run.out'
            output.write_text(text)
            log = ('LRI_R_PARTITION mode=atom_k rank=0 k1=4 k2=4\n'
                   + 'LRI_KMPI_FIELD cached=1 local_k=4 \n' * 6
                   + 'LRI_KMPI_TR_REUSE enabled=1 tested_pairs=1 accepted_pairs=1 accepted_max=1e-15\n' * 6)
            rank_log = work / 'libbse.d/libri_rank_0.log'
            rank_log.write_text(log)
            self.assertTrue(validate_kernel(work, 1, True, True, False, True)['passed'])
            for broken in [text.replace('1e-15', 'nan', 1),
                           text.replace('1e-15', '1e-2', 1),
                           text.replace('RI_REAL_SPACE_COMPARE', 'MISSING', 1),
                           text.replace('RI_KMPI_REFERENCE_GUARDS', 'MISSING')]:
                output.write_text(broken)
                with self.assertRaises(ValueError):
                    validate_kernel(work, 1, True, True, False, True)
            output.write_text(text)
            for broken in [log.replace('cached=1', 'cached=0'),
                           log.replace('accepted_pairs=1', 'accepted_pairs=0'),
                           log.replace('LRI_KMPI_FIELD', 'MISSING', 1)]:
                rank_log.write_text(broken)
                with self.assertRaises(ValueError):
                    validate_kernel(work, 1, True, True, False, True)


if __name__ == '__main__':
    unittest.main()

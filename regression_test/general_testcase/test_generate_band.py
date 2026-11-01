"""Check that generated segments cover each intended k point exactly once."""
# Author: Ruiyi Zhou.
import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'examples/08_fhiaims_fhiaims_libbse_interpolate_333_444/generate_band.py'
spec = importlib.util.spec_from_file_location('generate_band', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BandGridTests(unittest.TestCase):
    def test_complete_cubic_mesh(self):
        for n in (2, 3, 4, 8, 10, 14):
            points = []
            lines = list(module.band_lines(n))
            self.assertEqual(len(lines), n * n)
            for line in lines:
                words = line.split()
                self.assertEqual(words[:2], ['output', 'band'])
                self.assertEqual(int(words[8]), n)
                start = list(map(float, words[2:5]))
                stop = list(map(float, words[5:8]))
                for k in range(n):
                    point = [n * (a + (b - a) * k / (n - 1)) for a, b in zip(start, stop)]
                    self.assertTrue(all(abs(x - round(x)) < 1.e-12 for x in point))
                    points.append(tuple(round(x) for x in point))
            self.assertEqual(points, list(itertools.product(range(n), repeat=3)))

    def test_cli_and_invalid_grid(self):
        output = subprocess.check_output([sys.executable, str(SCRIPT), '8'], text=True)
        self.assertEqual(output.splitlines(), list(module.band_lines(8)))
        default = subprocess.check_output([sys.executable, str(SCRIPT)], text=True)
        self.assertEqual(default.splitlines(), list(module.band_lines(3)))
        for n in (0, 1):
            result = subprocess.run([sys.executable, str(SCRIPT), str(n)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        with self.assertRaises(ValueError):
            list(module.band_lines(1))


if __name__ == '__main__':
    unittest.main()

"""CI must fail when a configured test is absent, skipped, or malformed."""
# Author: Ruiyi Zhou.

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BACKEND = Path(__file__).resolve().parents[1] / 'backend'
sys.path.insert(0, str(BACKEND))
from ci_report import expected_suites, suite_report


class ReportTests(unittest.TestCase):
    def test_inventory_and_missing_new_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            plan = {'tests': [
                {'name': 'unit'}, {'name': 'new_unit'},
                {'name': 'optics', 'properties': [{'name': 'LABELS', 'value': ['real_si']}]},
                {'name': 'kernel', 'properties': [{'name': 'LABELS', 'value': ['lri_kernel']}]},
                {'name': 'external', 'properties': [{'name': 'LABELS', 'value': ['integration']}]},
            ]}
            (work / 'test-plan.json').write_text(json.dumps(plan))
            self.assertEqual(expected_suites(work / 'test-plan.json'),
                             (['unit', 'new_unit'], ['optics', 'kernel']))
            (work / 'real-si.xml').write_text('<testsuite><testcase name="optics"/>'
                                            '<testcase name="kernel"/></testsuite>')
            env = dict(os.environ, COMPILE_OUTCOME='success', GENERAL_OUTCOME='success',
                       PHYSICAL_OUTCOME='success')
            env.pop('GITHUB_STEP_SUMMARY', None)
            for extra, expected in [('', 1), ('<testcase name="new_unit"/>', 0)]:
                (work / 'regression.xml').write_text(
                    '<testsuite><testcase name="unit"/>' + extra + '</testsuite>')
                result = subprocess.run([sys.executable, str(BACKEND / 'ci_report.py'),
                                         '--build', str(work)], env=env, capture_output=True)
                self.assertEqual(result.returncode, expected, result.stdout.decode())
            (work / 'test-plan.json').unlink()
            result = subprocess.run([sys.executable, str(BACKEND / 'ci_report.py'),
                                     '--build', str(work)], env=env, capture_output=True)
            self.assertEqual(result.returncode, 1)

    def test_failures_are_not_green(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.xml'
            for xml in ('<bad', '<testsuite/>',
                        '<testsuite><testcase name="unit"><skipped/></testcase></testsuite>',
                        '<testsuite><testcase name="unit"><failure>bad</failure></testcase></testsuite>',
                        '<testsuite><testcase name="unit"/><testcase name="unit"/></testsuite>'):
                path.write_text(xml)
                self.assertFalse(suite_report(path, 'General', ['unit'], 'success')[1])
            path.unlink()
            self.assertFalse(suite_report(path, 'General', ['unit'], 'success')[1])


if __name__ == '__main__':
    unittest.main()

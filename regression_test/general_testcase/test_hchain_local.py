"""Check local H-chain preparation and refusal of incomplete/metallic SCF runs."""
# Author: Ruiyi Zhou.

import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[2] / "examples/09_hchain_local/run.py"
SPEC = importlib.util.spec_from_file_location("hchain_local", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        RUNNER.prepare(root / "scf", 8, "scf")
        control = (root / "scf/control.in").read_text()
        assert "k_grid 1 1 8\n" in control
        assert "qpe_calc" not in control
        assert "KS_method serial" in control
        assert (root / "scf/geometry.in").read_text().count("\natom ") == 2
        try:
            RUNNER.prepare(root / "scf", 4, "export")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing calculation was overwritten")
        assert (root / "scf/control.in").read_text() == control
        RUNNER.prepare(root / "export", 4, "export")
        assert "output librpa mommat" in (root / "export/control.in").read_text()
        try:
            RUNNER.prepare(root / "invalid", 0, "scf")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid k grid was accepted")
        assert not (root / "invalid").exists()
        baseline = root / "baseline"
        subprocess.run([sys.executable, str(SCRIPT), "--stage", "baseline", "--prepare-only",
                        "--work", str(baseline)], check=True, capture_output=True)
        assert "output librpa mommat" in (baseline / "electronic/control.in").read_text()
        assert "../electronic" in (baseline / "gw/librpa.in").read_text()
        assert "nvirt 4" in (baseline / "bse/libbse.in").read_text()
        output = root / "bse_output"
        output.mkdir()
        energy_file = output / "Excitation_Energy_singlet.dat"
        energy_file.write_text("1 2 3 4\n")
        (output / "oscillator_strength_singlet_tda.dat").write_text(
            "# header\n" + "0 1 1 1 1 1\n" * 4)
        (output / "spectrum_singlet_tda.dat").write_text("0 1 1 1 1\n")
        assert RUNNER.check_bse(output, 1) == 1
        for invalid in ["1 2 3", "1 2 3 nan", "-1 2 3 4"]:
            energy_file.write_text(invalid)
            try:
                RUNNER.check_bse(output, 1)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid BSE energies were accepted")
        energy_file.write_text("1 2 3 4")
        (output / "spectrum_singlet_tda.dat").write_text("0 nan 1 1 1\n")
        try:
            RUNNER.check_bse(output, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("nonfinite spectrum was accepted")

    completed = "Self-consistency cycle converged.\nHave a nice day.\n"
    assert RUNNER.check_scf(completed + "ESTIMATED overall HOMO-LUMO gap: 4.217 eV") == 4.217
    for output in [completed, completed + "Overall HOMO-LUMO gap: 0.0 eV",
                   "Overall HOMO-LUMO gap: 4.0 eV\nHave a nice day."]:
        try:
            RUNNER.check_scf(output)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid SCF result was accepted")


if __name__ == "__main__":
    main()

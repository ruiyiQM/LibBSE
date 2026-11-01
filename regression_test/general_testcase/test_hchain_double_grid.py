"""Check full-grid FHI packing, input preservation, and the solver comparison gate."""
# Author: Ruiyi Zhou.
import importlib.util
from pathlib import Path
import tempfile

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "examples/09_hchain_local/double_grid.py"
SPEC = importlib.util.spec_from_file_location("hchain_double_grid", SCRIPT)
GRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRID)


def fixture(root, nk):
    electronic = root / "electronic"
    electronic.mkdir(parents=True)
    (electronic / "geometry.in").write_text("lattice_vector 0 0 2.5\natom 0 0 0 H\n")
    (electronic / "basis_out").write_text("1 6 8 aims\n")
    (electronic / "velocity_matrix").write_bytes(b"fixture coarse velocity")
    with (electronic / "band_out").open("w") as output:
        output.write(f"{nk} 1 5 6 0\n")
        for k in range(nk):
            output.write(f"{k+1} 1\n")
            for band in range(5):
                output.write(f"{band+1} {2 if band == 0 else 0} {band*.1+k*.01} {band*2.7+k*.27}\n")
    with (electronic / "bz_sampling_out").open("w") as output:
        output.write(f"1 1 {nk}\n{nk} {nk}\n")
        for k in range(nk):
            output.write(f"{k+1} {1/nk} 0 0 {k/nk}\n")
    with (electronic / "KS_eigenvector_0.txt").open("w") as output:
        for k in range(nk):
            output.write(f"{k+1}\n")
            for ao in range(6):
                for band in range(5):
                    output.write(f"{100*k+10*ao+band} {-ao-.2*band}\n")
    (root / "gw/librpa.d").mkdir(parents=True)
    (root / "gw/energy_qp").write_text("fixture QP\n")
    (root / "gw/librpa.d/Wc_Mu_0_Nu_0_ifreq_0.mtx").write_text("fixture Wc\n")


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        coarse, fine, work = root / "coarse", root / "fine", root / "work"
        fixture(coarse, 2)
        fixture(fine, 4)
        before = {p: p.read_bytes() for source in [coarse, fine] for p in source.rglob("*") if p.is_file()}
        metadata = GRID.prepare(coarse, fine, work)
        assert metadata["coarse_nk"] == 2 and metadata["dense_nk"] == 4
        assert all(p.read_bytes() == content for p, content in before.items())
        packed = np.fromfile(work / "electronic/band_KS_eigenvector_k_00003.txt", dtype="<c16").reshape(5, 6)
        assert packed[3, 4] == complex(243, -4.6)
        table = np.loadtxt(work / "electronic/band_KS_eigenvalue_k_00002.txt")
        assert table[0, 2] == .5 and np.isclose(table[3, 3], .31)
        assert (work / "electronic/velocity_matrix").is_symlink()
        assert (work / "electronic/velocity_matrix").read_bytes() == b"fixture coarse velocity"
        fine3 = root / "fine3"
        fixture(fine3, 3)
        assert GRID.prepare(coarse, fine3, root / "noncommensurate")["dense_nk"] == 3
        assert "bse_solver haydock" in (work / "haydock/libbse.in").read_text()
        assert "out_bse_eigenvectors false" in (work / "haydock/libbse.in").read_text()
        try:
            GRID.prepare(coarse, fine, work)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing work was overwritten")
        (fine / "electronic/basis_out").write_text("different basis\n")
        try:
            GRID.prepare(coarse, fine, root / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("different AO bases were accepted")
        assert not (root / "bad").exists()
        spectrum = np.column_stack((np.arange(10)*.1, np.ones((10, 4))))
        for solver in ["elpa", "haydock"]:
            (work / solver / "libbse.d").mkdir()
            np.savetxt(work / solver / "libbse.d/spectrum_singlet_tda.dat", spectrum)
        assert GRID.compare(work)["pass"]
        for bad in [np.nan, 1.1]:
            spectrum[0, 2] = bad
            np.savetxt(work / "haydock/libbse.d/spectrum_singlet_tda.dat", spectrum)
            try:
                GRID.compare(work)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid spectrum passed the comparison gate")
            assert not (work / "comparison.json").exists()
        (fine / "electronic/KS_eigenvector_0.txt").write_text("1\nnan 0\n")
        try:
            GRID.read_export(fine / "electronic")
        except ValueError:
            pass
        else:
            raise AssertionError("incomplete/nonfinite KS export was accepted")


if __name__ == "__main__":
    main()

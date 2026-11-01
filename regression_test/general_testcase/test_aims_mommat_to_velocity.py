#!/usr/bin/env python3
"""Regression test for the FHI-aims Fortran HDF5 grid ordering."""

from pathlib import Path
import importlib.util
import struct
import tempfile

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "aims_mommat_to_velocity.py"
SPEC = importlib.util.spec_from_file_location("aims_mommat_to_velocity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def read_block(path: Path, ik: int, nk: int, nbands: int) -> np.ndarray:
    with path.open("rb") as stream:
        header = struct.unpack("<7i", stream.read(28))
        assert header == (-12345680, 29, nk, 1, nbands, nbands, 3)
        table = [struct.unpack("<iq", stream.read(12)) for _ in range(nk)]
        assert [item[0] for item in table] == list(range(1, nk + 1))
        stream.seek(table[ik][1])
        return np.fromfile(stream, dtype="<c16", count=3 * nbands * nbands).reshape(
            1, 3, nbands, nbands
        ) / CONVERTER.ATOMIC_TO_EV_ANGSTROM


def main() -> None:
    # h5py sees the Fortran-written grid as (kz, ky, kx) = (2, 1, 3).
    h5_shape = (2, 1, 3)
    canonical_shape = h5_shape[::-1]
    nk = int(np.prod(h5_shape))
    nbands = 3
    npairs = nbands * (nbands + 1) // 2
    with tempfile.TemporaryDirectory(prefix="libbse_mommat_test_") as temporary:
        directory = Path(temporary)
        (directory / "band_out").write_text(f"{nk}\n1\n{nbands}\n{nbands}\n")
        k_points = np.zeros((*h5_shape, 4))
        momentum = np.zeros((*h5_shape, npairs, 6))
        for kx, ky, kz in np.ndindex(canonical_shape):
            ik = (kx * canonical_shape[1] + ky) * canonical_shape[2] + kz
            source_cell = (kz, ky, kx)
            k_points[source_cell] = (ik, kx, ky, kz)
            for pair in range(npairs):
                for alpha in range(3):
                    base = 1000 * ik + 100 * alpha + pair + 1
                    momentum[source_cell + (pair, 2 * alpha)] = base
                    momentum[source_cell + (pair, 2 * alpha + 1)] = -base / 10
        with h5py.File(directory / "mommat.h5", "w") as handle:
            handle["Energy_window"] = np.array([[1.0], [3.0]])
            handle["k_points"] = k_points
            handle["Momentummatrix"] = momentum

        output = directory / "velocity_matrix"
        CONVERTER.convert(directory, output)

        # ik=4 lives at source HDF5 cell (kz,ky,kx)=(0,0,2). Pair 4 is
        # (state 2,state 3) in the row-major packed upper triangle.
        block = read_block(output, ik=4, nk=nk, nbands=nbands)
        gradient = complex(4000 + 4 + 1, -(4000 + 4 + 1) / 10)
        expected = -1j * gradient
        assert np.allclose(block[0, 0, 2, 1], expected)
        assert np.allclose(block[0, 0, 1, 2], np.conjugate(expected))

        # The native per-k stream must produce exactly the same canonical file.
        hdf5_reference = output.read_bytes()
        (directory / "mommat.h5").unlink()
        for kx, ky, kz in np.ndindex(canonical_shape):
            ik = (kx * canonical_shape[1] + ky) * canonical_shape[2] + kz
            packed = momentum[kz, ky, kx]
            # HDF5 packs upper gradients; the native stream packs lower ones.
            gradients = -(packed[:, 0::2] + 1j * packed[:, 1::2]).conjugate()
            payload = np.ascontiguousarray(gradients.T, dtype="<c16").tobytes()
            (directory / f"mommat_ks_kpt_{ik + 1:06d}.dat").write_bytes(
                struct.pack("<6i", ik + 1, 1, nbands, npairs, 1, 3) + payload)
        CONVERTER.convert(directory, output)
        assert output.read_bytes() == hdf5_reference

        # Use a physical anti-Hermitian gradient with complex band phases,
        # rather than only checking agreement between two hand-packed formats.
        rng = np.random.default_rng(193)
        coeff = rng.normal(size=(nbands, nbands)) + 1j * rng.normal(size=(nbands, nbands))
        ao = rng.normal(size=(3, nbands, nbands)) + 1j * rng.normal(size=(3, nbands, nbands))
        ao = ao + ao.swapaxes(-1, -2).conjugate()
        for ik in range(nk):
            gauged = np.exp(1j * (ik + 1) * np.arange(nbands))[:, None] * coeff
            conventional = gauged.conjugate() @ ao @ gauged.T
            # Native order is column outer, row >= column inner.
            rows, columns = np.triu_indices(nbands)
            native_gradient = (1j * conventional)[:, columns, rows]
            (directory / f"mommat_ks_kpt_{ik + 1:06d}.dat").write_bytes(
                struct.pack("<6i", ik + 1, 1, nbands, npairs, 1, 3)
                + np.ascontiguousarray(native_gradient, dtype="<c16").tobytes())
        CONVERTER.convert(directory, output)
        for ik in range(nk):
            gauged = np.exp(1j * (ik + 1) * np.arange(nbands))[:, None] * coeff
            stored = read_block(output, ik, nk, nbands)[0]
            inverse = np.linalg.inv(gauged)
            recovered = inverse.conjugate() @ stored.swapaxes(-1, -2) @ inverse.T
            assert np.allclose(recovered, ao, atol=1.e-11, rtol=1.e-11)

        # Preserve the independent physical payload for atomic-output checks.
        hdf5_reference = output.read_bytes()

        native = directory / "mommat_ks_kpt_000002.dat"
        original = native.read_bytes()
        for broken in [original[:12], original[:-16],
                       struct.pack("<6i", 99, 1, nbands, npairs, 1, 3) + original[24:],
                       struct.pack("<6i", 2, 1, nbands, npairs, 2, 3) + original[24:],
                       original[:24] + np.full(npairs * 3, np.nan, dtype="<c16").tobytes()]:
            native.write_bytes(broken)
            try:
                CONVERTER.convert(directory, output)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid native momentum input was accepted")
            assert output.read_bytes() == hdf5_reference
        native.write_bytes(original)
        native.unlink()
        try:
            CONVERTER.convert(directory, output)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing native k point was accepted")
        assert output.read_bytes() == hdf5_reference


if __name__ == "__main__":
    main()

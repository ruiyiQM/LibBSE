"""Checks for the experimental H-chain response interpolation control."""
# Author: Ruiyi Zhou.
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/09_hchain_local"))
import chi0_double_grid as control


def rejected(call, exception=ValueError):
    try:
        call()
    except exception:
        return
    raise AssertionError("invalid input accepted")


def main():
    a = np.array([[-3., .2], [.2, -4.]])
    b = np.array([[.3, .1], [.1, -.2]])
    c = np.array([[0., .1], [-.1, 0.]])
    for nc in (3, 4, 5, 8):
        chir = np.zeros((nc, 2, 2), complex)
        cells = np.arange(nc) - nc // 2
        chir[cells == 0] = a
        chir[cells == 1] = (b + c) / 2
        chir[cells == -1] = (b - c) / 2
        for nf in (7, 8, 13):
            q = 2 * np.pi * np.arange(nf) / nf
            exact = a + np.cos(q)[:, None, None] * b + 1j * np.sin(q)[:, None, None] * c
            assert np.allclose(control.interpolate_response(chir, nf), exact, atol=1e-14, rtol=0)
    chir = np.zeros((4, 2, 2), complex)
    chir[0] = b
    q = 2 * np.pi * np.arange(7) / 7
    assert np.allclose(control.interpolate_response(chir, 7), np.cos(2 * q)[:, None, None] * b)
    chir[2, 0, 0] = np.nan
    rejected(lambda: control.interpolate_response(chir, 7))

    response = np.tile(np.diag([-.2, -.1]), (3, 1, 1)).astype(complex)
    full = np.tile(np.diag([4., 9.]), (3, 1, 1))
    cut = np.tile(np.diag([1., 2.]), (3, 1, 1))
    gamma = np.diag([-.3, -.4])
    screened = control.screen_response(response, full, cut, gamma)
    assert np.array_equal(screened[0], gamma)
    expected = np.diag([1 / 1.8 - 1, 2 * (1 / 1.9 - 1)])
    assert np.allclose(screened[1:], expected, atol=1e-14, rtol=0)
    rejected(lambda: control.screen_response(-100 * response, full, cut, gamma))

    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        rejected(lambda: control.prepare(work / "absent", work / "absent", work), FileExistsError)
        file = work / "coulomb.bin"
        matrices = np.tile(np.eye(34, dtype=complex), (2, 1, 1))
        matrices[1, 0, 1] = 1 + 2j
        matrices[1, 1, 0] = 1 - 2j
        raw = struct.pack("<2i", 2, 2)
        for ik in (1, 2):
            raw += struct.pack("<6id", 34, 1, 34, 1, 34, ik, 1 / 3)
            raw += np.ascontiguousarray(matrices[ik - 1], dtype="<c16").tobytes()
        file.write_bytes(raw)
        result, original, gamma_region = control.read_coulomb(file, 3)
        assert original == raw and np.array_equal(result[2], matrices[1].conj())
        assert np.array_equal(np.frombuffer(raw[gamma_region], dtype="<c16").reshape(34, 34), matrices[0])
        for broken in (raw[:4], raw[:-16], raw + b"x", raw[:8] + struct.pack("<i", 33) + raw[12:]):
            file.write_bytes(broken)
            rejected(lambda: control.read_coulomb(file, 3))
        rejected(lambda: control.read_real_space(work, "Wc", 3))
        wcq = np.tile(np.eye(34, dtype=complex), (3, 1, 1))
        control.write_screening(work / "screening", wcq)
        wr = control.read_real_space(work / "screening", "Wc", 3)
        assert np.allclose(control.interpolate_response(wr, 3), wcq, atol=1e-13, rtol=0)
        file = work / "screening/Wc_Mu_0_Nu_0_iR_0_ifreq_0.mtx"
        lines = file.read_text().splitlines()
        lines[6] = lines[5]
        file.write_text("\n".join(lines) + "\n")
        rejected(lambda: control.read_real_space(work / "screening", "Wc", 3))


if __name__ == "__main__":
    main()

"""Member-level parallel ensembles are bit-identical to serial evaluation.

The cold-member policy makes ensemble members embarrassingly parallel: the
design is drawn once in the parent, every member solves cold in a fresh
forward model, and results are keyed by member index. This test pins the
contract that a process pool changes NOTHING numerically: same design, same
logliks, same predictions, same statuses, in the same order.
"""
import numpy as np
import pytest

from UQ.datasets import ChannelDNS

pytestmark = pytest.mark.skipif(
    not ChannelDNS.is_available(180),
    reason="channel DNS_data not present (bulk data is local/gitignored)",
)

# tiny, fast config sized for GENUINE convergence (same rationale as
# test_channel_baseline: measured ~5600 iterations on this mesh)
FAST_CFG = {"nx": 24, "ny": 32, "Lx": 10.0,
            "max_iter": 12000, "conv_tol": 1.0e-3, "yplus_target": 0.5}


def test_pool_matches_serial_bit_for_bit(rs):
    from UQ.datasets.channel_calibration import ChannelCalibration
    dns = ChannelDNS.load(180)

    c_serial = ChannelCalibration(dns, n_stations=8, cfg=FAST_CFG)
    nv_serial = c_serial.run_ensemble(n=3, seed=0)

    c_pool = ChannelCalibration(dns, n_stations=8, cfg=FAST_CFG)
    nv_pool = c_pool.run_ensemble(n=3, seed=0, n_workers=2)

    assert nv_pool == nv_serial
    np.testing.assert_array_equal(c_pool.X, c_serial.X)
    np.testing.assert_array_equal(c_pool.loglik, c_serial.loglik)
    np.testing.assert_array_equal(c_pool.preds, c_serial.preds)

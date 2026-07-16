"""RealNVP coupling-mask structure and exact invertibility.

Pins the audit fix to UQ.generative: coupling masks must alternate between
COMPLEMENTARY partitions at every layer for any target dimension (the old
i % dy start index shifted and shrank the mask for dy > 2, leaving late
layers transforming four of five anisotropy components conditioned on one),
and the flow must be exactly invertible with a consistent log-determinant
regardless of training state.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from UQ.generative import GenerativeDiscrepancyModel


def _masks(model):
    return [layer.mask.detach().numpy() for layer in model.flow.layers]


def test_masks_alternate_complementary_partitions_dy5():
    model = GenerativeDiscrepancyModel(n_features=3, n_targets=5,
                                       n_layers=6, hidden=16, seed=0)
    masks = _masks(model)
    assert len(masks) == 6
    for i in range(len(masks) - 1):
        # consecutive layers condition on complementary dims
        assert np.allclose(masks[i] + masks[i + 1], np.ones(5)), (i, masks[i])
    # every dim is transformed (mask == 0) in exactly half the layers
    transformed = sum((m == 0).astype(int) for m in masks)
    assert np.all(transformed == 3), transformed


def test_masks_unchanged_at_dy2():
    # the parity form coincides with the old indexing at dy = 2, so the
    # committed two-component (heat-flux) flows are architecturally unchanged
    model = GenerativeDiscrepancyModel(n_features=2, n_targets=2,
                                       n_layers=4, hidden=16, seed=0)
    masks = _masks(model)
    assert np.allclose(masks[0], [1.0, 0.0])
    assert np.allclose(masks[1], [0.0, 1.0])
    assert np.allclose(masks[2], [1.0, 0.0])
    assert np.allclose(masks[3], [0.0, 1.0])


def test_flow_round_trip_and_logdet_consistency():
    torch.manual_seed(0)
    model = GenerativeDiscrepancyModel(n_features=3, n_targets=5,
                                       n_layers=6, hidden=16, seed=1)
    flow = model.flow
    x = torch.randn(32, 5)
    ctx = torch.randn(32, 3)

    # data -> latent through the inverse chain, then back through forward
    z = x
    ld_inv = torch.zeros(32)
    for layer in reversed(flow.layers):
        z, ld = layer.inverse(z, ctx)
        ld_inv = ld_inv + ld
    x2 = z
    ld_fwd = torch.zeros(32)
    for layer in flow.layers:
        x2, ld = layer.forward(x2, ctx)
        ld_fwd = ld_fwd + ld

    assert torch.allclose(x2, x, atol=1e-4), "flow must invert exactly"
    # log-determinants of inverse and forward passes cancel
    assert torch.allclose(ld_inv + ld_fwd, torch.zeros(32), atol=1e-4)


def test_every_dim_reachable_from_latent():
    # with complementary masks, perturbing any latent dim moves the output:
    # no component is frozen by the mask pattern (the old dy=5 defect left
    # weakly coupled dims)
    torch.manual_seed(0)
    model = GenerativeDiscrepancyModel(n_features=2, n_targets=5,
                                       n_layers=6, hidden=16, seed=2)
    flow = model.flow
    ctx = torch.randn(1, 2)
    z0 = torch.zeros(1, 5)
    x0 = flow.push(z0, ctx)
    for j in range(5):
        z = z0.clone()
        z[0, j] = 1.0
        xj = flow.push(z, ctx)
        assert not torch.allclose(xj, x0, atol=1e-8), f"latent dim {j} inert"

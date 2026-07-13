"""Input contract for the bound Rhie-Chow checkerboard diagnostic."""

import numpy as np
import pytest


def test_odd_even_energy_ratio_requires_one_value_per_cell(rs):
    mesh = rs.Mesh.make_channel_2d(6, 4, 2.0, 1.0, Re=5000.0,
                                   yPlusTarget=1.0)
    n = mesh.n_cells()
    values = np.linspace(0.0, 1.0, n)
    assert np.isfinite(rs.odd_even_energy_ratio(mesh, list(values)))

    with pytest.raises(ValueError, match="exactly one value per cell"):
        rs.odd_even_energy_ratio(mesh, list(values[:-1]))
    with pytest.raises(ValueError, match="exactly one value per cell"):
        rs.odd_even_energy_ratio(mesh, list(np.append(values, 0.0)))

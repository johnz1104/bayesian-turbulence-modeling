"""The dict path of make_prior_from_param_set never touches the binding.

Pins the audit adjudication of the .pack duck-typing claim: the compiled
binding is imported lazily and only on the .pack branch, so pure-Python
(dict) use works under any interpreter regardless of binding ABI. The test
replaces the lazy accessor with one that fails loudly; building a prior from
a dict must succeed anyway.
"""
import numpy as np

import priors


def test_priors_dict_path_needs_no_binding(monkeypatch):
    def _forbidden():
        raise AssertionError("dict path must not import the compiled binding")

    monkeypatch.setattr(priors, "_rs", _forbidden)
    p = priors.make_prior_from_param_set(
        {"defaults": [0.31, 0.09], "lower": [0.2, 0.05], "upper": [0.5, 0.15]})
    assert np.allclose(p.means, [0.31, 0.09])
    assert p.lower.shape == (2,)


def test_unsupported_param_set_type_is_a_clear_error(rs):
    """Anything that is neither a Mapping nor the pybind parameter set is a
    contract violation reported as TypeError, not duck-typed into either
    branch."""
    import pytest
    from priors import make_prior_from_param_set
    with pytest.raises(TypeError):
        make_prior_from_param_set(42)

    class FakePack:
        def pack(self, c):
            return [0.0]
    with pytest.raises(TypeError):
        make_prior_from_param_set(FakePack())

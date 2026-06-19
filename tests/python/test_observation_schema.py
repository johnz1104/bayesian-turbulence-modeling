"""Tests for observation_schema.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from observation_schema import (
    CaseMetadata, CaseData, Observation, ObservationSet,
    GeometryType, WallCondition, DataSource, ObservableType,
    bfs_to_observation_set, channel_to_observation_set,
    koh_from_observation_set, scramjet_synthetic_observation_set,
)


# ---------------------------------------------------------------------------
# CaseMetadata
# ---------------------------------------------------------------------------

class TestCaseMetadata:
    def test_construction(self):
        m = CaseMetadata(
            case_id="test", geometry_type=GeometryType.CHANNEL,
            mach=0.1, reynolds=6800.0,
        )
        assert m.case_id == "test"
        assert m.mach == pytest.approx(0.1)
        assert m.wall_condition == WallCondition.ADIABATIC
        assert m.data_source == DataSource.SYNTHETIC

    def test_roundtrip_dict(self):
        m = CaseMetadata(
            case_id="bfs", geometry_type=GeometryType.BACKWARD_FACING_STEP,
            mach=0.0, reynolds=37600.0,
            stagnation_pressure=101325.0, stagnation_temperature=300.0,
            wall_condition=WallCondition.COOLED_WALL,
            data_source=DataSource.EXPERIMENT, notes="DS1985",
        )
        d = m.to_dict()
        m2 = CaseMetadata.from_dict(d)
        assert m2.case_id == m.case_id
        assert m2.geometry_type == m.geometry_type
        assert m2.wall_condition == m.wall_condition
        assert m2.data_source == m.data_source
        assert m2.stagnation_pressure == pytest.approx(m.stagnation_pressure)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TestObservation:
    def test_construction(self):
        o = Observation(
            observable_type=ObservableType.SKIN_FRICTION_CF,
            value=0.003, uncertainty=0.0001, x=5.0,
        )
        assert o.observable_type == ObservableType.SKIN_FRICTION_CF
        assert o.value == pytest.approx(0.003)
        assert o.x == pytest.approx(5.0)

    def test_roundtrip_dict(self):
        o = Observation(
            observable_type=ObservableType.WALL_HEAT_FLUX,
            value=50000.0, uncertainty=2500.0,
            x=0.5, y=0.0, units="W/m^2", group="qw_wall", case_id="test",
        )
        d = o.to_dict()
        o2 = Observation.from_dict(d)
        assert o2.observable_type == ObservableType.WALL_HEAT_FLUX
        assert o2.value == pytest.approx(50000.0)
        assert o2.units == "W/m^2"
        assert o2.group == "qw_wall"

    def test_observable_type_properties(self):
        assert ObservableType.SKIN_FRICTION_CF.is_wall_distributed
        assert ObservableType.WALL_PRESSURE_CP.is_wall_distributed
        assert ObservableType.WALL_HEAT_FLUX.is_wall_distributed
        assert ObservableType.MACH_PROFILE.is_profile
        assert ObservableType.SHOCK_LOCATION.is_scalar_location
        assert ObservableType.REATTACHMENT_LENGTH.is_scalar_location
        assert not ObservableType.SKIN_FRICTION_CF.is_profile


# ---------------------------------------------------------------------------
# ObservationSet
# ---------------------------------------------------------------------------

class TestObservationSet:
    @pytest.fixture
    def sample_obs_set(self):
        obs = ObservationSet("bfs")
        obs.add(Observation(ObservableType.REATTACHMENT_LENGTH, 6.26, 0.5, x=6.26))
        for x, cf, sig in [(2., 0.003, 0.0002), (5., 0.0028, 0.0002),
                           (8., 0.0031, 0.0002)]:
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF, cf, sig,
                                x=x, group="cf_wall"))
        return obs

    def test_n_obs(self, sample_obs_set):
        assert sample_obs_set.n_obs == 4

    def test_len(self, sample_obs_set):
        assert len(sample_obs_set) == 4

    def test_values_shape(self, sample_obs_set):
        v = sample_obs_set.values()
        assert v.shape == (4,)
        assert v[0] == pytest.approx(6.26)

    def test_uncertainties_positive(self, sample_obs_set):
        assert np.all(sample_obs_set.uncertainties() > 0)

    def test_locations_x(self, sample_obs_set):
        locs = sample_obs_set.locations_x()
        assert locs.shape == (4,)
        assert locs[0] == pytest.approx(6.26)

    def test_locations_xy_shape(self, sample_obs_set):
        xy = sample_obs_set.locations_xy()
        assert xy.shape == (4, 2)
        assert np.all(xy[:, 1] == 0.0)   # y defaults to 0

    def test_filter_by_type(self, sample_obs_set):
        cf_only = sample_obs_set.filter_by_type(ObservableType.SKIN_FRICTION_CF)
        assert cf_only.n_obs == 3
        xr_only = sample_obs_set.filter_by_type(ObservableType.REATTACHMENT_LENGTH)
        assert xr_only.n_obs == 1

    def test_filter_by_group(self, sample_obs_set):
        cf_wall = sample_obs_set.filter_by_group("cf_wall")
        assert cf_wall.n_obs == 3

    def test_filter_by_types(self, sample_obs_set):
        both = sample_obs_set.filter_by_types([
            ObservableType.SKIN_FRICTION_CF,
            ObservableType.REATTACHMENT_LENGTH,
        ])
        assert both.n_obs == 4

    def test_roundtrip_dict(self, sample_obs_set):
        d = sample_obs_set.to_dict()
        obs2 = ObservationSet.from_dict(d)
        assert obs2.case_id == sample_obs_set.case_id
        assert obs2.n_obs == sample_obs_set.n_obs
        assert np.allclose(obs2.values(), sample_obs_set.values())

    def test_roundtrip_json(self, sample_obs_set, tmp_path):
        path = tmp_path / "obs.json"
        sample_obs_set.to_json(path)
        obs2 = ObservationSet.from_json(path)
        assert obs2.n_obs == sample_obs_set.n_obs

    def test_to_koh_arrays_1d(self, sample_obs_set):
        locs, vals, sigs = sample_obs_set.to_koh_arrays(use_xy=False)
        assert locs.shape == (4,)
        assert vals.shape == (4,)
        assert sigs.shape == (4,)

    def test_to_koh_arrays_2d(self, sample_obs_set):
        locs, vals, sigs = sample_obs_set.to_koh_arrays(use_xy=True)
        assert locs.shape == (4, 2)

    def test_iter(self, sample_obs_set):
        for obs in sample_obs_set:
            assert isinstance(obs, Observation)

    def test_getitem(self, sample_obs_set):
        assert isinstance(sample_obs_set[0], Observation)

    def test_repr(self, sample_obs_set):
        r = repr(sample_obs_set)
        assert "bfs" in r


# ---------------------------------------------------------------------------
# CaseData
# ---------------------------------------------------------------------------

class TestCaseData:
    def test_roundtrip_json(self, tmp_path):
        meta = CaseMetadata("test", GeometryType.CHANNEL, 0.1, 6800.0)
        obs  = channel_to_observation_set(
            [5.0, 7.0, 9.0], [0.003, 0.003, 0.003], [0.0002] * 3, case_id="test"
        )
        cd = CaseData(meta, obs)
        path = tmp_path / "case.json"
        cd.to_json(path)
        cd2 = CaseData.from_json(path)
        assert cd2.case_id == "test"
        assert cd2.observations.n_obs == 3


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestBFSFactory:
    def test_bfs_observation_set(self):
        obs = bfs_to_observation_set(
            xr_h=6.26, xr_sigma=0.5,
            cf_stations=[2.0, 5.0, 8.0],
            cf_values=[0.003, 0.0028, 0.0031],
            cf_sigmas=[0.0002, 0.0002, 0.0002],
        )
        assert obs.n_obs == 4
        types = set(obs.types())
        assert ObservableType.REATTACHMENT_LENGTH in types
        assert ObservableType.SKIN_FRICTION_CF in types

    def test_bfs_koh_arrays(self):
        obs = bfs_to_observation_set(6.26, 0.5, [2., 5., 8.],
                                     [0.003, 0.0028, 0.0031], [0.0002]*3)
        locs, vals, sigs = obs.to_koh_arrays()
        assert locs.shape == (4,)
        assert np.all(sigs > 0)

    def test_channel_observation_set(self):
        obs = channel_to_observation_set(
            [5.0, 7.0, 9.0], [0.003, 0.003, 0.003], [0.0002] * 3
        )
        assert obs.n_obs == 3
        assert all(t == ObservableType.SKIN_FRICTION_CF for t in obs.types())


# ---------------------------------------------------------------------------
# KOH integration
# ---------------------------------------------------------------------------

class TestKOHFromObservationSet:
    def test_diagonal_mode(self):
        obs = bfs_to_observation_set(6.26, 0.5, [2., 5.], [0.003, 0.003],
                                     [0.0002, 0.0002])
        koh = koh_from_observation_set(obs, mode="diagonal")
        assert koh.n == 3

    def test_physical_gp_mode(self):
        obs = channel_to_observation_set([5., 7., 9.], [0.003]*3, [0.0002]*3)
        koh = koh_from_observation_set(obs, mode="physical_gp")
        assert koh.n == 3

    def test_koh_evaluates_finite(self):
        obs = channel_to_observation_set([5., 7.], [0.003, 0.003], [0.0002, 0.0002])
        koh = koh_from_observation_set(obs, mode="diagonal")
        eta = np.array([0.003, 0.003])
        ll  = koh(eta, log_sigma_delta=-3.0, log_l_delta=0.0)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# Scramjet synthetic generator
# ---------------------------------------------------------------------------

class TestScramjetSynthetic:
    def test_returns_expected_types(self):
        obs_set, truth, metadata = scramjet_synthetic_observation_set(
            n_wall_stations=5, mach=2.0
        )
        assert isinstance(obs_set, ObservationSet)
        assert isinstance(truth, dict)
        assert isinstance(metadata, CaseMetadata)

    def test_n_observations(self):
        obs_set, _, _ = scramjet_synthetic_observation_set(n_wall_stations=5)
        # 3 obs per station (Cp, Cf, qw) + 2 scalar locations
        assert obs_set.n_obs == 5 * 3 + 2

    def test_observable_types_present(self):
        obs_set, _, _ = scramjet_synthetic_observation_set(n_wall_stations=4)
        types = set(obs_set.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types
        assert ObservableType.SHOCK_LOCATION   in types
        assert ObservableType.REATTACHMENT_LOCATION in types

    def test_uncertainties_positive(self):
        obs_set, _, _ = scramjet_synthetic_observation_set(n_wall_stations=6)
        assert np.all(obs_set.uncertainties() > 0)

    def test_truth_keys(self):
        _, truth, _ = scramjet_synthetic_observation_set()
        for k in ("a1", "betaStar", "x_shock", "x_sep", "x_reat", "mach", "reynolds"):
            assert k in truth

    def test_metadata_geometry(self):
        _, _, meta = scramjet_synthetic_observation_set(mach=2.0)
        assert meta.geometry_type == GeometryType.SCRAMJET_INLET
        assert meta.mach == pytest.approx(2.0)
        assert meta.data_source == DataSource.SYNTHETIC

    def test_reproducible_with_seed(self):
        obs1, t1, _ = scramjet_synthetic_observation_set(rng_seed=0)
        obs2, t2, _ = scramjet_synthetic_observation_set(rng_seed=0)
        assert np.allclose(obs1.values(), obs2.values())
        assert t1["a1"] == pytest.approx(t2["a1"])

    def test_different_seeds_differ(self):
        obs1, _, _ = scramjet_synthetic_observation_set(rng_seed=0)
        obs2, _, _ = scramjet_synthetic_observation_set(rng_seed=999)
        assert not np.allclose(obs1.values(), obs2.values())


# ---------------------------------------------------------------------------
# ObservationSet semantics, mixed sets, validation
# ---------------------------------------------------------------------------

class TestObservationSetOrdering:
    """Observations must be returned in insertion order."""

    def test_insertion_order_preserved_values(self):
        obs = ObservationSet("ord_test")
        vals = [0.003, 0.0028, 0.0031, 0.006]
        for i, v in enumerate(vals):
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF, v, 0.0002,
                                x=float(i)))
        assert list(obs.values()) == pytest.approx(vals)

    def test_insertion_order_preserved_types(self):
        obs = ObservationSet("ord2")
        types_in = [ObservableType.SKIN_FRICTION_CF,
                    ObservableType.WALL_PRESSURE_CP,
                    ObservableType.WALL_HEAT_FLUX,
                    ObservableType.SHOCK_LOCATION]
        for t in types_in:
            obs.add(Observation(t, 1.0, 0.1))
        assert obs.types() == types_in

    def test_json_roundtrip_preserves_order(self, tmp_path):
        obs = ObservationSet("rt_test")
        xs = [0.0, 0.5, 1.0]
        for x in xs:
            obs.add(Observation(ObservableType.WALL_PRESSURE_CP, 0.5, 0.025, x=x))
        path = tmp_path / "rt.json"
        obs.to_json(path)
        obs2 = ObservationSet.from_json(path)
        assert list(obs2.locations_x()) == pytest.approx(xs)


class TestMixedObservableSets:
    """Tests for sets containing multiple observable types."""

    @pytest.fixture
    def cp_cf_set(self):
        obs = ObservationSet("mixed_cp_cf")
        xs = np.linspace(0, 1, 6)
        for x in xs:
            obs.add(Observation(ObservableType.WALL_PRESSURE_CP,
                                0.5 * x, 0.025 * x + 1e-4, x=float(x),
                                group="cp"))
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF,
                                0.002, 0.0001, x=float(x), group="cf"))
        return obs

    @pytest.fixture
    def cp_cf_qw_set(self):
        obs = ObservationSet("mixed_all3")
        for x in np.linspace(0, 1, 5):
            obs.add(Observation(ObservableType.WALL_PRESSURE_CP,
                                0.4, 0.02, x=float(x)))
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF,
                                0.002, 0.0001, x=float(x)))
            obs.add(Observation(ObservableType.WALL_HEAT_FLUX,
                                60000.0, 3000.0, x=float(x)))
        return obs

    def test_cp_cf_n_obs(self, cp_cf_set):
        assert cp_cf_set.n_obs == 12

    def test_filter_cp_from_mixed(self, cp_cf_set):
        cp_only = cp_cf_set.filter_by_type(ObservableType.WALL_PRESSURE_CP)
        assert cp_only.n_obs == 6
        assert all(t == ObservableType.WALL_PRESSURE_CP for t in cp_only.types())

    def test_filter_cf_from_mixed(self, cp_cf_set):
        cf_only = cp_cf_set.filter_by_type(ObservableType.SKIN_FRICTION_CF)
        assert cf_only.n_obs == 6

    def test_filter_by_group_cp(self, cp_cf_set):
        cp_group = cp_cf_set.filter_by_group("cp")
        assert cp_group.n_obs == 6

    def test_all_three_types_present(self, cp_cf_qw_set):
        types = set(cp_cf_qw_set.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types

    def test_filter_qw_from_all_three(self, cp_cf_qw_set):
        qw = cp_cf_qw_set.filter_by_type(ObservableType.WALL_HEAT_FLUX)
        assert qw.n_obs == 5
        assert np.all(qw.uncertainties() > 0)

    def test_koh_arrays_from_mixed_set(self, cp_cf_set):
        locs, vals, sigs = cp_cf_set.to_koh_arrays(use_xy=False)
        assert locs.shape == (12,)
        assert vals.shape == (12,)
        assert sigs.shape == (12,)
        assert np.all(sigs > 0)

    def test_filter_by_types_two_types(self, cp_cf_qw_set):
        subset = cp_cf_qw_set.filter_by_types([
            ObservableType.WALL_PRESSURE_CP,
            ObservableType.SKIN_FRICTION_CF,
        ])
        assert subset.n_obs == 10   # 5 Cp + 5 Cf

    def test_add_all(self):
        obs = ObservationSet("addall_test")
        extra = [
            Observation(ObservableType.SKIN_FRICTION_CF, 0.002, 0.0001),
            Observation(ObservableType.WALL_PRESSURE_CP, 0.5,   0.025),
        ]
        obs.add_all(extra)
        assert obs.n_obs == 2


class TestObservationSetMetadata:
    """Provenance and metadata preservation."""

    def test_case_id_propagated(self):
        obs = ObservationSet("my_case")
        obs.add(Observation(ObservableType.SKIN_FRICTION_CF, 0.002, 0.0001,
                            case_id="my_case"))
        d = obs.to_dict()
        obs2 = ObservationSet.from_dict(d)
        assert obs2.case_id == "my_case"
        assert obs2[0].case_id == "my_case"

    def test_units_preserved_json(self, tmp_path):
        obs = ObservationSet("units_test")
        obs.add(Observation(ObservableType.WALL_HEAT_FLUX, 60000.0, 3000.0,
                            units="W/m^2", group="qw_wall", case_id="units_test"))
        path = tmp_path / "units.json"
        obs.to_json(path)
        obs2 = ObservationSet.from_json(path)
        assert obs2[0].units == "W/m^2"
        assert obs2[0].group == "qw_wall"

    def test_reference_quantities_preserved(self, tmp_path):
        obs = ObservationSet("ref_test")
        o = Observation(ObservableType.SKIN_FRICTION_CF, 0.002, 0.0001,
                        reference_length=0.127, reference_velocity=50.0,
                        reference_pressure=101325.0, case_id="ref_test")
        obs.add(o)
        path = tmp_path / "ref.json"
        obs.to_json(path)
        obs2 = ObservationSet.from_json(path)
        assert obs2[0].reference_length == pytest.approx(0.127)
        assert obs2[0].reference_velocity == pytest.approx(50.0)
        assert obs2[0].reference_pressure == pytest.approx(101325.0)


class TestObservationSetValidation:
    """Observations with bad values should produce informative errors or be detectable."""

    def test_values_array_is_finite(self):
        obs = ObservationSet("fin_test")
        for v in [0.001, 0.002, 0.003]:
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF, v, 0.0001))
        assert np.all(np.isfinite(obs.values()))

    def test_uncertainties_are_positive(self):
        obs = ObservationSet("pos_unc")
        for sig in [0.0001, 0.0002, 0.0003]:
            obs.add(Observation(ObservableType.WALL_PRESSURE_CP, 0.5, sig))
        assert np.all(obs.uncertainties() > 0)

    def test_locations_x_shape(self):
        obs = ObservationSet("loc_shape")
        for i in range(5):
            obs.add(Observation(ObservableType.SKIN_FRICTION_CF, 0.002, 0.0001,
                                x=float(i) * 0.25))
        xs = obs.locations_x()
        assert xs.shape == (5,)
        assert xs.ndim == 1

    def test_locations_xy_shape(self):
        obs = ObservationSet("loc_xy")
        for i in range(4):
            obs.add(Observation(ObservableType.MACH_PROFILE, 2.0, 0.05,
                                x=0.5, y=float(i) * 0.01))
        xy = obs.locations_xy()
        assert xy.shape == (4, 2)
        assert np.all(np.isfinite(xy))

    def test_empty_obs_set(self):
        obs = ObservationSet("empty")
        assert obs.n_obs == 0
        assert len(obs.values()) == 0
        assert len(obs.uncertainties()) == 0

    def test_json_roundtrip_fixture(self):
        import json
        from pathlib import Path
        fixture = (Path(__file__).parent.parent
                   / "fixtures" / "high_mach" / "synthetic_sbli_casedata.json")
        cd = CaseData.from_json(fixture)
        assert cd.metadata.mach == pytest.approx(2.0)
        assert cd.metadata.data_source == DataSource.SYNTHETIC
        assert cd.observations.n_obs == 13
        # Ensure filter by type works on loaded fixture
        cp_obs = cd.observations.filter_by_type(ObservableType.WALL_PRESSURE_CP)
        assert cp_obs.n_obs == 5

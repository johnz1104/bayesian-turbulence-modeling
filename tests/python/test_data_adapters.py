"""Tests for data_adapters.py."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_adapters import (
    CSVLoader, JSONLoader, NPZLoader, load_wall_data,
    make_synthetic_high_mach_npz, make_synthetic_high_mach_csv,
)
from observation_schema import ObservationSet, ObservableType


# ---------------------------------------------------------------------------
# Fixtures — temp files with synthetic data
# ---------------------------------------------------------------------------

@pytest.fixture()
def csv_wall_file(tmp_path):
    """CSV with x, cp, cp_sigma, cf, cf_sigma, qw, qw_sigma columns."""
    p = tmp_path / "wall.csv"
    return make_synthetic_high_mach_csv(p, n_stations=8, rng_seed=0)


@pytest.fixture()
def npz_wall_file(tmp_path):
    """NPZ with x, cp, cp_sigma, cf, cf_sigma, qw, qw_sigma arrays."""
    p = tmp_path / "wall.npz"
    return make_synthetic_high_mach_npz(p, n_stations=8, rng_seed=0)


@pytest.fixture()
def json_col_file(tmp_path):
    """JSON column-oriented file."""
    xs = list(np.linspace(0, 1, 6).tolist())
    data = {
        "x":        xs,
        "cf":       [0.003 + 0.0001 * i for i in range(6)],
        "cf_sigma": [0.0002] * 6,
        "qw":       [50000.0 + 1000 * i for i in range(6)],
    }
    p = tmp_path / "data.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def json_obsset_file(tmp_path):
    """JSON ObservationSet dict format."""
    obs_set = ObservationSet("ext_case")
    from observation_schema import Observation
    for i in range(4):
        obs_set.add(Observation(ObservableType.SKIN_FRICTION_CF,
                                value=0.003, uncertainty=0.0002,
                                x=float(i), case_id="ext_case"))
    p = tmp_path / "obsset.json"
    obs_set.to_json(p)
    return p


# ---------------------------------------------------------------------------
# CSVLoader
# ---------------------------------------------------------------------------

class TestCSVLoader:
    def test_loads_correct_n_obs(self, csv_wall_file):
        obs = CSVLoader().load(csv_wall_file, case_id="csv_test")
        # 8 stations × 3 observables (cp, cf, qw)
        assert obs.n_obs == 24

    def test_all_uncertainties_positive(self, csv_wall_file):
        obs = CSVLoader().load(csv_wall_file)
        assert np.all(obs.uncertainties() > 0)

    def test_observable_types(self, csv_wall_file):
        obs = CSVLoader().load(csv_wall_file)
        types = set(obs.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types

    def test_x_locations_match_csv(self, csv_wall_file):
        obs = CSVLoader().load(csv_wall_file)
        xs = obs.locations_x()
        assert xs[0] == pytest.approx(0.0, abs=1e-6)
        assert xs[-1] == pytest.approx(1.0, abs=0.2)

    def test_case_id_set(self, csv_wall_file):
        obs = CSVLoader().load(csv_wall_file, case_id="my_case")
        assert obs.case_id == "my_case"

    def test_default_uncertainty_fallback(self, tmp_path):
        """CSV without sigma columns → fallback to frac×|value|."""
        p = tmp_path / "no_sigma.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "cf"])
            for i in range(5):
                w.writerow([str(float(i)), "0.003"])
        obs = CSVLoader().load(p, default_uncertainty_frac=0.10)
        assert np.all(obs.uncertainties() > 0)


# ---------------------------------------------------------------------------
# JSONLoader
# ---------------------------------------------------------------------------

class TestJSONLoader:
    def test_column_oriented(self, json_col_file):
        obs = JSONLoader().load(json_col_file, case_id="json_col")
        # 6 stations × 2 recognized observables (cf, qw)
        assert obs.n_obs == 12
        types = set(obs.types())
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types

    def test_obsset_format(self, json_obsset_file):
        obs = JSONLoader().load(json_obsset_file, case_id="override_ignored")
        assert obs.n_obs == 4
        assert obs.case_id == "ext_case"   # original case_id preserved

    def test_list_of_obs_dicts(self, tmp_path):
        from observation_schema import Observation
        obs_list = [Observation(ObservableType.SKIN_FRICTION_CF,
                                0.003, 0.0002, x=float(i)).to_dict()
                    for i in range(3)]
        p = tmp_path / "list.json"
        p.write_text(json.dumps(obs_list))
        obs = JSONLoader().load(p, case_id="list_case")
        assert obs.n_obs == 3

    def test_full_casedata_format(self, tmp_path):
        from observation_schema import CaseData, CaseMetadata, GeometryType, DataSource
        meta = CaseMetadata("cd_case", GeometryType.CHANNEL, 0.1, 6800.0)
        obs_raw = ObservationSet("cd_case")
        from observation_schema import Observation
        obs_raw.add(Observation(ObservableType.SKIN_FRICTION_CF, 0.003, 0.0002))
        cd = CaseData(meta, obs_raw)
        p = tmp_path / "casedata.json"
        cd.to_json(p)
        obs = JSONLoader().load(p)
        assert obs.n_obs == 1

    def test_uncertainties_positive(self, json_col_file):
        obs = JSONLoader().load(json_col_file)
        assert np.all(obs.uncertainties() > 0)


# ---------------------------------------------------------------------------
# NPZLoader
# ---------------------------------------------------------------------------

class TestNPZLoader:
    def test_loads_correct_n_obs(self, npz_wall_file):
        obs = NPZLoader().load(npz_wall_file, case_id="npz_test")
        assert obs.n_obs == 24   # 8 stations × 3 obs

    def test_uncertainties_from_sigma_arrays(self, npz_wall_file):
        obs = NPZLoader().load(npz_wall_file)
        sigs = obs.uncertainties()
        assert np.all(sigs > 0)

    def test_observable_types(self, npz_wall_file):
        obs = NPZLoader().load(npz_wall_file)
        types = set(obs.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types

    def test_x_locations(self, npz_wall_file):
        obs = NPZLoader().load(npz_wall_file)
        xs = obs.locations_x()
        assert xs.min() == pytest.approx(0.0, abs=1e-6)
        assert xs.max() == pytest.approx(1.0, abs=0.2)

    def test_no_sigma_fallback(self, tmp_path):
        p = tmp_path / "nosig.npz"
        np.savez(str(p), x=np.linspace(0, 1, 5), cf=np.full(5, 0.003))
        obs = NPZLoader().load(p, default_uncertainty_frac=0.10)
        assert obs.n_obs == 5
        assert np.all(obs.uncertainties() > 0)


# ---------------------------------------------------------------------------
# Auto-detect load_wall_data
# ---------------------------------------------------------------------------

class TestAutoDetect:
    def test_csv(self, csv_wall_file):
        obs = load_wall_data(csv_wall_file)
        assert obs.n_obs > 0

    def test_json(self, json_col_file):
        obs = load_wall_data(json_col_file)
        assert obs.n_obs > 0

    def test_npz(self, npz_wall_file):
        obs = load_wall_data(npz_wall_file)
        assert obs.n_obs > 0

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="unsupported extension"):
            load_wall_data(p)


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------

class TestSyntheticGenerators:
    def test_npz_created(self, tmp_path):
        p = make_synthetic_high_mach_npz(tmp_path / "syn.npz", n_stations=10)
        assert p.exists()

    def test_npz_round_trip(self, tmp_path):
        p = make_synthetic_high_mach_npz(tmp_path / "syn.npz", n_stations=10,
                                          rng_seed=42)
        obs = load_wall_data(p, case_id="syn")
        assert obs.n_obs == 30  # 10 × 3

    def test_csv_created(self, tmp_path):
        p = make_synthetic_high_mach_csv(tmp_path / "syn.csv", n_stations=8)
        assert p.exists()

    def test_csv_round_trip(self, tmp_path):
        p = make_synthetic_high_mach_csv(tmp_path / "syn.csv", n_stations=8)
        obs = load_wall_data(p)
        assert obs.n_obs == 24  # 8 × 3

    def test_reproducible_with_seed(self, tmp_path):
        p1 = make_synthetic_high_mach_npz(tmp_path / "a.npz", rng_seed=0)
        p2 = make_synthetic_high_mach_npz(tmp_path / "b.npz", rng_seed=0)
        o1 = NPZLoader().load(p1)
        o2 = NPZLoader().load(p2)
        assert np.allclose(o1.values(), o2.values())

    def test_koh_pipeline_accepts_loaded_data(self, npz_wall_file):
        obs = load_wall_data(npz_wall_file, case_id="koh_test")
        koh = obs.filter_by_type(ObservableType.SKIN_FRICTION_CF)
        from observation_schema import koh_from_observation_set
        likelihood = koh_from_observation_set(koh, mode="diagonal")
        assert likelihood.n == koh.n_obs
        eta = koh.values() * 1.01   # slightly wrong prediction
        ll  = likelihood(eta, log_sigma_delta=-3.0, log_l_delta=0.0)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# Alias column tests
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "high_mach"


class TestColumnAliases:
    """Verify that non-canonical column names are recognised."""

    def test_skin_friction_alias(self, tmp_path):
        p = tmp_path / "alias_cf.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "skin_friction", "skin_friction_sigma"])
            for i in range(5):
                w.writerow([str(float(i) * 0.25), "0.0021", "0.0001"])
        obs = load_wall_data(p)
        assert obs.n_obs == 5
        assert all(t == ObservableType.SKIN_FRICTION_CF for t in obs.types())

    def test_pressure_coefficient_alias(self, tmp_path):
        p = tmp_path / "alias_cp.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "pressure_coefficient"])
            for i in range(4):
                w.writerow([str(float(i) * 0.33), "0.5"])
        obs = load_wall_data(p)
        assert obs.n_obs == 4
        assert all(t == ObservableType.WALL_PRESSURE_CP for t in obs.types())

    def test_heat_flux_alias(self, tmp_path):
        p = tmp_path / "alias_qw.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "heat_flux"])
            for i in range(3):
                w.writerow([str(float(i) * 0.5), "50000.0"])
        obs = load_wall_data(p)
        assert obs.n_obs == 3
        assert all(t == ObservableType.WALL_HEAT_FLUX for t in obs.types())

    def test_mach_uppercase_alias(self, tmp_path):
        """Column named 'M' (capital) should map to MACH_PROFILE."""
        p = tmp_path / "alias_M.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "M"])
            for i in range(4):
                w.writerow([str(float(i) * 0.25), f"{1.8 + 0.1 * i}"])
        obs = load_wall_data(p)
        assert obs.n_obs == 4
        assert all(t == ObservableType.MACH_PROFILE for t in obs.types())

    def test_x_over_h_location_alias(self, tmp_path):
        """x_over_h should be recognised as the location column."""
        p = tmp_path / "alias_xloc.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_over_h", "cf", "cf_sigma"])
            for i in range(5):
                w.writerow([str(float(i)), "0.002", "0.0001"])
        obs = load_wall_data(p)
        assert obs.n_obs == 5
        xs = obs.locations_x()
        assert xs[0] == pytest.approx(0.0, abs=1e-6)
        assert xs[-1] == pytest.approx(4.0, abs=1e-6)

    def test_x_mm_location_alias(self, tmp_path):
        """x_mm should be recognised as the location column."""
        p = tmp_path / "alias_xmm.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x_mm", "cp"])
            for i in range(4):
                w.writerow([str(float(i) * 10.0), "0.5"])
        obs = load_wall_data(p)
        assert obs.n_obs == 4
        xs = obs.locations_x()
        assert xs[-1] == pytest.approx(30.0, abs=0.1)


# ---------------------------------------------------------------------------
# Fixture-based tests
# ---------------------------------------------------------------------------

class TestHighMachFixtures:
    """Load the committed fixture files and verify round-trip correctness."""

    def test_wall_pressure_only_csv(self):
        p = FIXTURE_DIR / "wall_pressure_only.csv"
        obs = load_wall_data(p, case_id="fixture_cp")
        assert obs.n_obs == 15
        assert all(t == ObservableType.WALL_PRESSURE_CP for t in obs.types())
        assert np.all(obs.uncertainties() > 0)
        xs = obs.locations_x()
        assert xs[0] == pytest.approx(0.0, abs=1e-6)
        assert xs[-1] == pytest.approx(1.0, abs=0.01)

    def test_skin_friction_only_csv(self):
        p = FIXTURE_DIR / "skin_friction_only.csv"
        obs = load_wall_data(p, case_id="fixture_cf")
        assert obs.n_obs == 15
        assert all(t == ObservableType.SKIN_FRICTION_CF for t in obs.types())

    def test_heat_flux_only_csv(self):
        p = FIXTURE_DIR / "heat_flux_only.csv"
        obs = load_wall_data(p, case_id="fixture_qw")
        assert obs.n_obs == 15
        assert all(t == ObservableType.WALL_HEAT_FLUX for t in obs.types())

    def test_cp_cf_combined_csv(self):
        p = FIXTURE_DIR / "cp_cf_combined.csv"
        obs = load_wall_data(p, case_id="fixture_cp_cf")
        assert obs.n_obs == 16   # 8 stations × 2 observables
        types = set(obs.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types

    def test_cp_cf_qw_combined_csv(self):
        p = FIXTURE_DIR / "cp_cf_qw_combined.csv"
        obs = load_wall_data(p, case_id="fixture_all3")
        assert obs.n_obs == 24   # 8 stations × 3 observables
        types = set(obs.types())
        assert ObservableType.WALL_HEAT_FLUX in types

    def test_json_casedata_fixture(self):
        p = FIXTURE_DIR / "synthetic_sbli_casedata.json"
        obs = load_wall_data(p, case_id="fixture_json")
        assert obs.n_obs > 0
        types = set(obs.types())
        assert ObservableType.WALL_PRESSURE_CP in types
        assert ObservableType.SKIN_FRICTION_CF in types
        assert ObservableType.WALL_HEAT_FLUX   in types

    def test_npz_fixture(self):
        p = FIXTURE_DIR / "synthetic_high_mach.npz"
        obs = load_wall_data(p, case_id="fixture_npz")
        assert obs.n_obs == 45   # 15 stations × 3 obs
        assert np.all(obs.uncertainties() > 0)


# ---------------------------------------------------------------------------
# Malformed-file tests
# ---------------------------------------------------------------------------

class TestMalformedInputs:
    def test_csv_no_recognised_observable_columns_raises(self, tmp_path):
        """CSV with unrecognised column names should raise ValueError."""
        p = tmp_path / "bad_cols.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "unknown_quantity1", "unknown_quantity2"])
            w.writerow(["0.0", "1.0", "2.0"])
        with pytest.raises(ValueError, match="no recognisable observable"):
            load_wall_data(p)

    def test_csv_bad_sigma_falls_back_gracefully(self):
        """CSV with non-numeric sigma value → fallback to fractional default."""
        p = FIXTURE_DIR / "malformed_bad_sigma.csv"
        obs = load_wall_data(p, case_id="bad_sigma")
        assert obs.n_obs == 3
        assert np.all(obs.uncertainties() > 0)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.dat"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="unsupported extension"):
            load_wall_data(p)

    def test_json_unrecognised_format_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(42))  # int at top level
        with pytest.raises((ValueError, AttributeError, TypeError)):
            load_wall_data(p)

    def test_csv_with_missing_x_column_uses_zero(self):
        """CSV whose only non-observable column is station_id (no x) loads with x=0."""
        p = FIXTURE_DIR / "malformed_missing_x.csv"
        # 'station_id' is not a recognised x-location column → x defaults to 0
        # cp and cf ARE recognised, so we get observations with x=0
        obs = load_wall_data(p, case_id="no_x")
        assert obs.n_obs > 0
        assert np.all(obs.locations_x() == 0.0)


# ---------------------------------------------------------------------------
# Duplicate / unsorted x locations
# ---------------------------------------------------------------------------

class TestUnsortedDuplicateX:
    def test_unsorted_x_loads_ok(self, tmp_path):
        """Unsorted x values should load without error."""
        p = tmp_path / "unsorted.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "cf"])
            for x in [0.9, 0.1, 0.5, 0.3, 0.7]:
                w.writerow([str(x), "0.002"])
        obs = load_wall_data(p)
        assert obs.n_obs == 5
        xs = obs.locations_x()
        assert set(xs) == {0.9, 0.1, 0.5, 0.3, 0.7}

    def test_duplicate_x_loads_ok(self, tmp_path):
        """Duplicate x values are valid (multiple sensor channels at same station)."""
        p = tmp_path / "dup_x.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "cf"])
            for _ in range(3):
                w.writerow(["0.5", "0.002"])
        obs = load_wall_data(p)
        assert obs.n_obs == 3
        assert np.all(obs.locations_x() == pytest.approx(0.5))

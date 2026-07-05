"""Compressible plane-channel DNS loader (Gerolymos and Vallet 2023).

Parses the Gerolymos-Vallet isothermal-wall compressible turbulent plane-channel
matrix (J. Fluid Mech. 958 (2023) A19, doi:10.1017/jfm.2023.42; distribution
Mendeley Data doi:10.17632/wt8t5kxzbs.1, CC BY 4.0; flow model AIR0 per J. Fluid
Mech. 757 (2014) 701-746) into the canonical compressible dns_field record
(UQ.datasets._compressible: Favre moments, wall units). This is the workhorse
calibration matrix of the compressible attached-flow study: 24 cases spanning
HCB friction Reynolds number Re_tau* 97 to 985 and centreline Mach number
M_CLx 0.32 to 2.49, isothermal walls at 298 K.

Raw layout, per case, under DNS_data/compressible_channel_gv/GV_TPC_MB_AIR0/
<case>/ with <case> = Retaus_NNNN_MCLx_MpMM_isoTw_0298_MB_AIR0 and file prefix
GV_TPC_NNNN_MpMM_isoTw_0298_MB_AIR0:

  0_GD_global_data/*_GD_global_data.txt   42 tabulated global parameters
      (Re_tau*, M_CLx, Re_tau_w, the wall-heat-flux parameter B_qw, M_tau, cf,
      gamma, molecular Prandtl numbers); the wall-flux source.
  1_PBs_profiles_and_budgets/*_MF_meanflow.txt   50 mean-profile columns
      (wall-distance in y/delta, y*, y#, y+ conventions; Reynolds and Favre
      means side by side; density, temperature, viscosity, conductivity, Mach,
      molecular-Prandtl profiles).
  1_PBs_profiles_and_budgets/*_TRBFLXs_turbulent_transport.txt   98 columns,
      of which this loader consumes the FAVRE REYNOLDS-STRESS TENSOR
      <rho u_i"u_j">* (columns 5 to 10), the FAVRE TURBULENT ENTHALPY-FLUX
      VECTOR <rho h"u_i">* (columns 72 to 74), and the unit-conversion columns
      <rho>/<rho_w> (97) and <u_CL>/V_unit*(y) (98).
  (budgets_r*.txt, TTS, rms_skew_flat, Vlct_HoMs, 2_pdfsq, 3_pdfs2q are present
   but not consumed by this loader.)

Every file header states lines_of_comments / columns_of_data / lines_of_data
and a per-column label block ending in one consolidated "label |N|" line; the
loader parses the header counts and the label map and NEVER hardcodes column
positions. Unit conversions applied here (derivations in the code comments):

  stress:        <rho u_i"u_j">* = <rho u_i"u_j">/tau_w = <rho u_i"u_j">^+
                 R_ij = <rho u_i"u_j">^+ / rho^+                [velocity^2 +]
  enthalpy flux: <rho h"u_i">* = <rho h"u_i">/sqrt(tau_w^3/<rho>)
                 q_hat_i = <rho h"u_i">*/(sqrt(rho^+) rho^+) (gamma_w-1) M_tau^2
                 (the Favre temperature flux in (u_tau, T_w) units)

The physics anchor is the variable-density total-stress balance
mu^+ dU^+/dy^+ - <rho u"v">^+ = 1 - y/delta (exact for the fully-developed
channel), whose interior rms is the data's own convergence level
(UQ.datasets.observation_sigma).
"""
import os
import re

import numpy as np

from . import _common
from ._compressible import CompressibleProfileDNS

# the 24 case directories present in the compiled local copy (the Mendeley
# distribution states 25 conditions; the count delta is recorded in
# DNS_data/README.md). Directory name encodes nominal Re_tau* and M_CLx.
GV_CASES = (
    "Retaus_0097_MCLx_2p11_isoTw_0298_MB_AIR0",
    "Retaus_0098_MCLx_2p22_isoTw_0298_MB_AIR0",
    "Retaus_0099_MCLx_2p01_isoTw_0298_MB_AIR0",
    "Retaus_0100_MCLx_1p82_isoTw_0298_MB_AIR0",
    "Retaus_0100_MCLx_1p92_isoTw_0298_MB_AIR0",
    "Retaus_0103_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0105_MCLx_0p32_isoTw_0298_MB_AIR0",
    "Retaus_0106_MCLx_0p79_isoTw_0298_MB_AIR0",
    "Retaus_0112_MCLx_2p02_isoTw_0298_MB_AIR0",
    "Retaus_0113_MCLx_2p49_isoTw_0298_MB_AIR0",
    "Retaus_0114_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0134_MCLx_1p99_isoTw_0298_MB_AIR0",
    "Retaus_0143_MCLx_0p32_isoTw_0298_MB_AIR0",
    "Retaus_0151_MCLx_0p79_isoTw_0298_MB_AIR0",
    "Retaus_0151_MCLx_1p50_isoTw_0298_MB_AIR0",
    "Retaus_0177_MCLx_0p35_isoTw_0298_MB_AIR0",
    "Retaus_0245_MCLx_1p99_isoTw_0298_MB_AIR0",
    "Retaus_0251_MCLx_0p83_isoTw_0298_MB_AIR0",
    "Retaus_0254_MCLx_1p47_isoTw_0298_MB_AIR0",
    "Retaus_0340_MCLx_0p80_isoTw_0298_MB_AIR0",
    "Retaus_0341_MCLx_1p98_isoTw_0298_MB_AIR0",
    "Retaus_0342_MCLx_1p51_isoTw_0298_MB_AIR0",
    "Retaus_0965_MCLx_1p50_isoTw_0298_MB_AIR0",
    "Retaus_0985_MCLx_0p81_isoTw_0298_MB_AIR0",
)

_SUBDIR = os.path.join("compressible_channel_gv", "GV_TPC_MB_AIR0")

# "label |N|" pairs on the consolidated header line (the last comment line)
_LABEL = re.compile(r"([^|]+?)\s*\|\s*(\d+)\|")

# "|N| ... = value" pairs in the GD description block (the header states every
# global inline as well as tabulating it; the loader test asserts both agree)
_GD_INLINE = re.compile(r"\|\s*(\d+)\|[^=]*=\s*(-?[0-9.]+(?:[eE][+-]?[0-9]+)?)")

# the simulation-data block repeated in every header (loader-test cross-check)
_SIM_PATTERNS = {
    "m_clx": re.compile(r"<M_CL_x>\s*=\s*(-?[0-9.eE+-]+)"),
    "re_tau_star": re.compile(r"Re_tau\*\s*=\s*(-?[0-9.eE+-]+)"),
    "re_tau_w": re.compile(r"Re_tau_w\s*=\s*(-?[0-9.eE+-]+)"),
}

# consumed column labels, exactly as printed in the consolidated header lines
# (including the source's own quirks: the FA-temperature label's "{T>" typo and
# the trailing space inside "<gamma_w >")
_MF_COLS = {
    "y_outer": "y/delta",
    "ystar": "y*",
    "yplus": "y+",
    "rho": "<rho>/<rho>_w",
    "T_reynolds": "<T>/<T>_w",
    "mach": "<M_x>",
    "mu": "<mu>/<mu>_w",
    "lambda_plus": "<lmbd>/<lmbd>_w",
    "T": "{T>/{T}_w",
    "U_reynolds": "<u>+",
    "U": "{u}+",
    "pr_molecular": "(<mu> <cp>)/<lmbd>",
}
_TF_COLS = {
    "yplus": "y+",
    "rho_uu": '<rho u"u">*',
    "rho_uv": '<rho u"v">*',
    "rho_vv": '<rho v"v">*',
    "rho_vw": '<rho v"w">*',
    "rho_ww": '<rho w"w">*',
    "rho_wu": '<rho w"u">*',
    "rho_hu": '<rho h"u">*',
    "rho_hv": '<rho h"v">*',
    "rho_hw": '<rho h"w">*',
    "rho_plus": "<rho>/<rho_w>",
    "ucl_over_vunit": "<u_CL>/V_unit*(y)",
}
_GD_COLS = {
    "re_tau_star": "Re_tau*",
    "m_clx": "<M_CL_x>",
    "re_tau_w": "Re_tau_w",
    "b_q": "B_qw",
    "m_tau": "M_tau",
    "cf": "cf",
    "gamma_w": "<gamma_w >",
    "pr_w": "Pr_w",
}


class GVChannelDNS(CompressibleProfileDNS):
    """One Gerolymos-Vallet compressible channel case as the canonical record.

    In addition to the compressible base record (yplus, U, R, k, re_tau, T,
    rho, mu, ystar, q_hat, mach, pr_molecular, wall):

      U_reynolds, T_reynolds   the Reynolds-averaged mean views (cross-checks;
                               the record's U and T are the Favre means)
      lambda_plus              mean heat conductivity on its wall value
      ucl_over_vunit           <u_CL>/V_unit*(y), the file's own *-to-outer
                               velocity-scale conversion column
      gd                       all 42 tabulated global parameters, by label
      sim                      the header's simulation-data block (Re_tau*,
                               M_CLx, Re_tau_w), for consistency tests
    """

    def __init__(self, case, mf, tf, gd, sim, meta):
        n = mf["yplus"].size
        if tf["yplus"].size != n:
            raise ValueError(f"{case}: meanflow has {n} stations but "
                             f"turbulent-transport has {tf['yplus'].size}")
        if not np.allclose(mf["yplus"], tf["yplus"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{case}: meanflow and turbulent-transport y+ "
                             "grids differ")

        # Favre stress, velocity^2 wall-unit form: the file's *-unit stress is
        # already the momentum-flux + form (<rho u_i"u_j">* = <rho u_i"u_j">/
        # tau_w = <rho u_i"u_j">+), so dividing by <rho>+ gives
        # R_ij = <rho u_i"u_j">/(<rho> u_tau^2)
        rho_plus = tf["rho_plus"]
        R = _common.assemble_tensor(
            tf["rho_uu"] / rho_plus, tf["rho_vv"] / rho_plus,
            tf["rho_ww"] / rho_plus, tf["rho_uv"] / rho_plus,
            uw=tf["rho_wu"] / rho_plus, vw=tf["rho_vw"] / rho_plus)
        k = 0.5 * np.trace(R, axis1=1, axis2=2)

        # Favre temperature flux in (u_tau, T_w) units. Derivation:
        #   <rho h"u">* := <rho h"u">/sqrt(tau_w^3/<rho>)
        #     => <rho h"u">/(rho_w u_tau^3) = <rho h"u">*/sqrt(rho+)
        #   q_hat = <rho h"u">/(<rho> cp_w u_tau T_w)
        #         = <rho h"u">*/(sqrt(rho+) rho+) * u_tau^2/(cp_w T_w)
        #   u_tau^2/(cp_w T_w) = (gamma_w - 1) M_tau^2     [cp T = a^2/(gamma-1)]
        # AIR0 has constant cp, so cp_w is THE cp and no profile ratio enters.
        scale = (gd["gamma_w"] - 1.0) * gd["m_tau"] ** 2 \
            / (np.sqrt(rho_plus) * rho_plus)
        q_hat = np.stack([tf["rho_hu"] * scale, tf["rho_hv"] * scale,
                          tf["rho_hw"] * scale], axis=-1)

        wall = {
            "b_q": gd["b_q"], "m_tau": gd["m_tau"], "cf": gd["cf"],
            "re_tau_star": gd["re_tau_star"], "gamma_w": gd["gamma_w"],
            "pr_w": gd["pr_w"], "m_clx": gd["m_clx"],
        }
        super().__init__(
            yplus=mf["yplus"], U=mf["U"], R=R, k=k, re_tau=gd["re_tau_w"],
            meta=meta, T=mf["T"], rho=mf["rho"], mu=mf["mu"],
            y_outer=mf["y_outer"], ystar=mf["ystar"], q_hat=q_hat,
            mach=mf["mach"], pr_molecular=mf["pr_molecular"], wall=wall)
        self.case = case
        self.U_reynolds = mf["U_reynolds"]
        self.T_reynolds = mf["T_reynolds"]
        self.lambda_plus = mf["lambda_plus"]
        self.ucl_over_vunit = tf["ucl_over_vunit"]
        self.gd = gd["all"]
        self.sim = sim

    # ---- location -----------------------------------------------------------

    @staticmethod
    def case_dir(case, root=None):
        return os.path.join(_common.data_root(root), _SUBDIR, case)

    @staticmethod
    def is_available(case, root=None):
        return os.path.isdir(GVChannelDNS.case_dir(case, root))

    @staticmethod
    def parse_tag(case):
        """Nominal (Re_tau*, M_CLx) encoded in the case directory name."""
        m = re.match(r"Retaus_(\d+)_MCLx_(\d+)p(\d+)_", case)
        if m is None:
            raise ValueError(f"not a GV case directory name: {case}")
        return int(m.group(1)), float(f"{m.group(2)}.{m.group(3)}")

    @staticmethod
    def _paths(case, root=None):
        base = GVChannelDNS.case_dir(case, root)
        prefix = case.replace("Retaus_", "GV_TPC_").replace("_MCLx", "")
        return (
            os.path.join(base, "0_GD_global_data",
                         prefix + "_GD_global_data.txt"),
            os.path.join(base, "1_PBs_profiles_and_budgets",
                         prefix + "_MF_meanflow.txt"),
            os.path.join(base, "1_PBs_profiles_and_budgets",
                         prefix + "_TRBFLXs_turbulent_transport.txt"),
        )

    # ---- header-driven parsing ----------------------------------------------

    @staticmethod
    def _read_table(path):
        """Parse one GV file: header counts, label map, simulation block, data.

        Line 3 of every file states lines_of_comments, columns_of_data and
        lines_of_data; the LAST comment line is the consolidated "label |N|"
        map. Both are authoritative and the data block is validated against
        them, so no column position or count is ever hardcoded.
        """
        with open(path) as fh:
            head = [fh.readline() for _ in range(3)]
            counts = head[2].split()
            n_comments, n_cols, n_rows = (int(counts[1]), int(counts[2]),
                                          int(counts[3]))
            rest = [fh.readline() for _ in range(n_comments - 3)]
        header = head + rest
        labels = {}
        for name, col in _LABEL.findall(header[-1].lstrip("# ")):
            labels[name.strip()] = int(col) - 1
        if len(labels) != n_cols:
            raise ValueError(f"{path}: consolidated header line has "
                             f"{len(labels)} labels, expected {n_cols}")
        sim = {}
        text = "".join(header)
        for key, pattern in _SIM_PATTERNS.items():
            m = pattern.search(text)
            if m is not None:
                sim[key] = float(m.group(1))
        data = np.atleast_2d(np.loadtxt(path, skiprows=n_comments))
        if data.shape != (n_rows, n_cols):
            raise ValueError(f"{path}: data block is {data.shape}, header "
                             f"states ({n_rows}, {n_cols})")
        return labels, sim, data, header

    @staticmethod
    def _columns(path, wanted):
        """Load the wanted columns of one profile file, by header label."""
        labels, sim, data, _ = GVChannelDNS._read_table(path)
        out = {}
        for key, label in wanted.items():
            if label not in labels:
                raise ValueError(f"{path}: column '{label}' not in header")
            out[key] = data[:, labels[label]]
        return out, sim

    @staticmethod
    def _globals(path):
        """Parse the GD file: the 42 tabulated globals plus the inline copies.

        The GD header restates every tabulated value inline ("... = value");
        both are parsed and returned so the loader test can assert the file is
        self-consistent.
        """
        labels, sim, data, header = GVChannelDNS._read_table(path)
        row = data[0]
        gd = {key: float(row[labels[label]])
              for key, label in _GD_COLS.items()}
        gd["all"] = {label: float(row[col]) for label, col in labels.items()}
        inline = {}
        for line in header:
            for col, value in _GD_INLINE.findall(line):
                inline.setdefault(int(col) - 1, float(value))
        gd["inline"] = inline
        gd["sim"] = sim
        return gd

    # ---- loading -------------------------------------------------------------

    @staticmethod
    def load(case, root=None):
        """Parse one case (by directory name) into the canonical record."""
        gd_path, mf_path, tf_path = GVChannelDNS._paths(case, root)
        gd = GVChannelDNS._globals(gd_path)
        mf, sim = GVChannelDNS._columns(mf_path, _MF_COLS)
        tf, _ = GVChannelDNS._columns(tf_path, _TF_COLS)
        meta = {
            "regime": "compressible",
            "case": "gv_compressible_channel",
            "averaging": "favre",
            "source": "Gerolymos and Vallet 2023 (JFM 958 A19); "
                      "Mendeley doi:10.17632/wt8t5kxzbs.1 (CC BY 4.0)",
            "case_dir": case,
            "wall_thermal": "isothermal 298 K",
            "sigma_note": "modeled observation uncertainty "
                          "(no per-point statistical uncertainty in file)",
        }
        return GVChannelDNS(case=case, mf=mf, tf=tf, gd=gd, sim=sim, meta=meta)

    @staticmethod
    def load_all(root=None):
        """Load every available case, ordered as in GV_CASES (Re_tau* major)."""
        return [GVChannelDNS.load(c, root) for c in GV_CASES
                if GVChannelDNS.is_available(c, root)]

    # ---- physics anchor -------------------------------------------------------

    def total_stress_target(self):
        """The GV channel is driven per unit MASS, pinned by the data itself.

        Against the uniform-force target 1 - y/delta the outer-region balance
        is off by O(M^2) percent with the density-defect shape (3 percent at
        M_CLx 2.49); against the per-unit-mass (density-weighted) target it
        closes to 0.03 percent there, two orders better. The remaining
        buffer-layer residual is the neglected viscosity-fluctuation
        correlation <mu'(du/dy)'> absent from the file's columns, which is why
        the observation-sigma anchor masks the buffer layer out.
        """
        return self.total_stress_target_mass_forced()

    def __repr__(self):
        return (f"GVChannelDNS(Re_tau*={self.wall['re_tau_star']:.0f}, "
                f"M_CLx={self.wall['m_clx']:.2f}, n={self.n})")

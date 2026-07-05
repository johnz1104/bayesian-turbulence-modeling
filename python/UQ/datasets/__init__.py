"""Raw-DNS loaders that emit the canonical dns_field record.

Each real dataset gets one loader module here. Every loader parses its own raw
format but returns the SAME canonical record (DNS_data/README.md, "Standardized
processing format"), so the discrepancy, UQ, and evaluation layers downstream
always see one uniform representation. The channel loader (Lee and Moser 2015)
is the first and establishes the pattern (location, naming, provenance) every
later dataset reuses (Couette, separated, compressible, SBLI).
"""
from .channel import ChannelDNS, CHANNEL_CASES
from .couette import CouetteDNS, COUETTE_CASES
from .pipe import PipeDNS, PIPE_CASES
from .rotating_channel import RotatingChannelDNS, ROTATING_CASES
from .backward_facing_step import BackwardFacingStepDNS, BFS_STATIONS
from .periodic_hills import PeriodicHillsDNS, PEHILL_CASES
from .gv_channel import GVChannelDNS, GV_CASES
from .zdc_flatplate import FlatPlateDNS, ZDC_CASES
from .ckm_channel import CKMChannelDNS, CKM_CASES

__all__ = [
    "ChannelDNS", "CHANNEL_CASES",
    "CouetteDNS", "COUETTE_CASES",
    "PipeDNS", "PIPE_CASES",
    "RotatingChannelDNS", "ROTATING_CASES",
    "BackwardFacingStepDNS", "BFS_STATIONS",
    "PeriodicHillsDNS", "PEHILL_CASES",
    "GVChannelDNS", "GV_CASES",
    "FlatPlateDNS", "ZDC_CASES",
    "CKMChannelDNS", "CKM_CASES",
]

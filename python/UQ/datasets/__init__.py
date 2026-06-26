"""Raw-DNS loaders that emit the canonical dns_field record.

Each real dataset gets one loader module here. Every loader parses its own raw
format but returns the SAME canonical record (DNS_data/README.md, "Standardized
processing format"), so the discrepancy, UQ, and evaluation layers downstream
always see one uniform representation. The channel loader (Lee and Moser 2015)
is the first and establishes the pattern (location, naming, provenance) every
later dataset reuses (Couette, separated, compressible, SBLI).
"""
from .channel import ChannelDNS, CHANNEL_CASES

__all__ = ["ChannelDNS", "CHANNEL_CASES"]

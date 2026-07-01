# Thread B: hypersonic SBLI application

Application thread for the research direction. This thread addresses hypersonic
separated flows (shock-boundary-layer interaction), where the deterministic,
local, memoryless Boussinesq closure is least reliable, using a streamline-history
non-local stochastic closure. The current solver is low-Mach (Ma about 0.5 and
below), so the high-speed regime is the intended application, not a present capability;
work here builds toward it.

Any experiment placed here requires parameters in a config file (never hard-coded
in logic), a reproduce script, and fixed random seeds, plus a provenance entry in
data/README.md for any dataset it consumes. There is no real hypersonic / SBLI
data in this repository; any high-Mach case must state its source or mark itself
clearly synthetic. See the root CLAUDE.md for the full conventions.

# Thread A: chaotic-systems methodology

Methodology proving ground for the research direction. This thread develops and
validates the stochastic, non-local closure machinery (a generalized Langevin
equation closure: Markovian drift plus memory integral plus stochastic forcing,
derived from the Mori-Zwanzig formalism) together with its Bayesian uncertainty
quantification on canonical chaotic systems (for example Kuramoto-Sivashinsky,
Burgers, Lorenz-96) before any of it is applied to flow. These systems are cheap,
exactly reproducible, and expose memory and non-locality cleanly.

Any experiment placed here requires parameters in a config file (never hard-coded
in logic), a reproduce script, and fixed random seeds, plus a provenance entry in
data/README.md for any dataset it consumes. See the root CLAUDE.md for the full
conventions.

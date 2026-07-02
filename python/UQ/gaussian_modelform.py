"""Gaussian (Kennedy-O'Hagan-type) model-form baseline over the anisotropy
discrepancy.

The second comparison baseline of the separated-flow model-form study: a
heteroscedastic Gaussian conditional model over the same five independent
anisotropy-discrepancy components, conditioned on the same five invariant
features as the generative model, and pushed through the same realizability
projection and the same Reynolds-stress injection. It differs from the
generative model in exactly one respect, the distribution family (a diagonal
Gaussian versus a normalizing flow), which is the property under test: a
Gaussian cannot represent the skewed, heteroscedastic, possibly multimodal
discrepancy law, and the comparison measures what that costs. The projection
grants the Gaussian samples realizability they do not have by construction,
which is charitable to this baseline and noted where reported (see
UQ-RANS_research/separated_modelform/METHODS_OPERATIONALIZATION.md).

The API mirrors ``generative.GenerativeDiscrepancyModel`` (fit / sample /
log_prob / sample_realizable_anisotropy) so the study drives both through the
same code path. PyTorch is imported lazily, exactly as in ``generative``.
"""
import numpy as np

from . import realizability as rz
from .generative import GenerativeDiscrepancyModel, _MLP


def _torch():
    import torch  # lazy: only needed when the model is actually used
    return torch


class GaussianDiscrepancyModel:
    """Heteroscedastic diagonal-Gaussian conditional discrepancy model.

    p(y | x) = N(mu(x), diag(exp(logvar(x)))) with mu and logvar small MLPs,
    fit by maximum likelihood on (features, targets) pairs.
    """

    def __init__(self, n_features, n_targets, hidden=64, seed=0):
        torch = _torch()
        torch.manual_seed(seed)
        self.dy = n_targets
        self.dc = n_features
        self.mean_net = _MLP.build(n_features, n_targets, hidden)
        self.logvar_net = _MLP.build(n_features, n_targets, hidden)
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

    def _standardise_fit(self, X, Y):
        self._x_mean, self._x_std = X.mean(0), X.std(0) + 1e-8
        self._y_mean, self._y_std = Y.mean(0), Y.std(0) + 1e-8

    def fit(self, features, targets, epochs=400, lr=1e-3, batch=256, verbose=False):
        torch = _torch()
        X = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
        Y = np.asarray(targets, dtype=np.float32).reshape(len(targets), -1)
        self._standardise_fit(X, Y)
        Xs = torch.tensor((X - self._x_mean) / self._x_std)
        Ys = torch.tensor((Y - self._y_mean) / self._y_std)
        params = list(self.mean_net.parameters()) + list(self.logvar_net.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        n = len(Xs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                mu = self.mean_net(Xs[idx])
                logv = torch.clamp(self.logvar_net(Xs[idx]), -12.0, 6.0)
                # Gaussian negative log-likelihood per point:
                #   0.5 [ (y - mu)^2 / exp(logv) + logv + log(2 pi) ]
                nll = 0.5 * (((Ys[idx] - mu) ** 2) * torch.exp(-logv)
                             + logv + np.log(2.0 * np.pi))
                loss = nll.sum(-1).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if verbose and ep % 50 == 0:
                print(f"epoch {ep}  nll {tot / n:.4f}")
        return self

    def _mu_sigma(self, features):
        torch = _torch()
        X = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
        Xs = torch.tensor((X - self._x_mean) / self._x_std)
        with torch.no_grad():
            mu = self.mean_net(Xs).numpy()
            logv = np.clip(self.logvar_net(Xs).numpy(), -12.0, 6.0)
        sigma = np.exp(0.5 * logv)
        return mu, sigma

    def sample(self, features, n_per=1):
        """Draw n_per Gaussian samples per feature row. Returns (N, n_per, dy).

        Uses the torch generator (seeded at construction), exactly as the
        generative flow does, so fixed-seed reproduce scripts govern both.
        """
        torch = _torch()
        mu, sigma = self._mu_sigma(features)
        with torch.no_grad():
            eps = torch.randn(mu.shape[0], n_per, self.dy).numpy()
        ys = mu[:, None, :] + sigma[:, None, :] * eps
        return ys * self._y_std + self._y_mean

    def log_prob(self, features, targets):
        mu, sigma = self._mu_sigma(features)
        Y = np.asarray(targets, dtype=np.float32).reshape(len(targets), -1)
        Ys = (Y - self._y_mean) / self._y_std
        z = (Ys - mu) / sigma
        lp = -0.5 * (z ** 2 + np.log(2.0 * np.pi)) - np.log(sigma)
        # change of variables for the target standardisation
        return lp.sum(-1) - np.log(self._y_std).sum()

    @staticmethod
    def components_to_anisotropy(comp):
        """Five independent components to a symmetric traceless 3x3 batch."""
        return GenerativeDiscrepancyModel.components_to_anisotropy(comp)

    def sample_realizable_anisotropy(self, features, base_anisotropy=None, n_per=1):
        """Sample, add the base anisotropy, and project into the realizable set.

        The projection is the same barycentric projection every method's
        samples pass through before injection; without it Gaussian draws leave
        the realizable set (the known Kennedy-O'Hagan deficiency).
        """
        comp = self.sample(features, n_per=n_per)             # (N, n_per, 5)
        b = self.components_to_anisotropy(comp)               # (N, n_per, 3, 3)
        if base_anisotropy is not None:
            b = b + base_anisotropy[:, None, :, :]
        bp, _ = rz.project_anisotropy(b)
        return bp

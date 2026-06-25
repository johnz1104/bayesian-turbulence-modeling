"""Generative, realizability-constrained model-form model.

A conditional normalizing flow (RealNVP) over the model-form discrepancy,
conditioned on the Galilean-invariant features, with samples projected into the
barycentric realizable set. This is the flagship technique of
compressible_research.md (N4): unlike a Gaussian Kennedy-O'Hagan term or a
deterministic correction it represents a non-Gaussian, heteroscedastic, possibly
multimodal discrepancy law, which is what the discrepancy near a shock foot
actually looks like, and unlike an eigenspace-perturbation envelope every sample
is a genuine draw that is strictly realizable.

The flow is a universal conditional density approximator (Papamakarios et al.
2021); realizability is imposed afterwards by ``realizability.project_anisotropy``
so the two constraints (invariance via the features, realizability via the
projection) stay separate, as the project requires.

PyTorch is imported lazily so importing the rest of the ``UQ`` package never
requires it; the generative test skips when torch is absent.
"""
import numpy as np

from . import realizability as rz


def _torch():
    import torch  # lazy: only needed when the generative model is actually used
    return torch


class _MLP:
    """Factory for a small tanh MLP (kept as a staticmethod-style helper)."""
    @staticmethod
    def build(n_in, n_out, hidden):
        torch = _torch()
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_out),
        )


def _make_flow(dy, dc, n_layers, hidden):
    """Build the conditional RealNVP module (defined lazily so torch is optional)."""
    torch = _torch()
    import torch.nn as nn

    class Coupling(nn.Module):
        # Affine coupling conditioned on the context; `mask` keeps part of y
        # fixed and transforms the rest from the fixed part plus the context.
        def __init__(self, mask):
            super().__init__()
            self.register_buffer("mask", mask)
            self.scale = _MLP.build(dy + dc, dy, hidden)
            self.trans = _MLP.build(dy + dc, dy, hidden)
            self.scale_scale = nn.Parameter(torch.zeros(dy))

        def _st(self, y, ctx):
            ym = y * self.mask
            h = torch.cat([ym, ctx], dim=-1)
            s = torch.tanh(self.scale(h)) * self.scale_scale  # bounded log-scale
            t = self.trans(h)
            s = s * (1 - self.mask)
            t = t * (1 - self.mask)
            return s, t

        def forward(self, z, ctx):                  # latent -> data
            s, t = self._st(z, ctx)
            x = z * torch.exp(s) + t
            return x, s.sum(-1)

        def inverse(self, x, ctx):                  # data -> latent
            s, t = self._st(x, ctx)
            z = (x - t) * torch.exp(-s)
            return z, -s.sum(-1)

    class RealNVP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for i in range(n_layers):
                m = torch.zeros(dy)
                m[i % dy::2] = 1.0 if dy > 1 else 1.0
                if dy == 1:                          # alternate a 1-D mask trivially
                    m = torch.tensor([float(i % 2)])
                layers.append(Coupling(m))
            self.layers = nn.ModuleList(layers)

        def log_prob(self, x, ctx):
            z = x
            logdet = torch.zeros(x.shape[0])
            for layer in reversed(self.layers):
                z, ld = layer.inverse(z, ctx)
                logdet = logdet + ld
            base = -0.5 * (z ** 2).sum(-1) - 0.5 * dy * np.log(2 * np.pi)
            return base + logdet

        def sample(self, ctx):
            z = torch.randn(ctx.shape[0], dy)
            x = z
            for layer in self.layers:
                x, _ = layer.forward(x, ctx)
            return x

    return RealNVP()


class GenerativeDiscrepancyModel:
    """Conditional normalizing-flow model over a model-form discrepancy.

    fit on (features, targets); sample or score p(targets | features). The
    targets are whatever discrepancy parametrisation is supplied; for a Reynolds
    anisotropy discrepancy use the five independent components and the
    ``sample_realizable_anisotropy`` helper to project draws into the realizable
    set.
    """

    def __init__(self, n_features, n_targets, n_layers=8, hidden=64, seed=0):
        torch = _torch()
        torch.manual_seed(seed)
        self.dy = n_targets
        self.dc = n_features
        self.flow = _make_flow(n_targets, n_features, n_layers, hidden)
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
        opt = torch.optim.Adam(self.flow.parameters(), lr=lr)
        n = len(Xs)
        for ep in range(epochs):
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = -self.flow.log_prob(Ys[idx], Xs[idx]).mean()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            if verbose and ep % 50 == 0:
                print(f"epoch {ep}  nll {tot / n:.4f}")
        return self

    def sample(self, features, n_per=1):
        """Draw n_per samples of the target per feature row. Returns (N, n_per, dy)."""
        torch = _torch()
        X = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
        Xs = (X - self._x_mean) / self._x_std
        out = np.zeros((len(X), n_per, self.dy), dtype=np.float32)
        with torch.no_grad():
            for j in range(n_per):
                ctx = torch.tensor(Xs)
                ys = self.flow.sample(ctx).numpy()
                out[:, j, :] = ys * self._y_std + self._y_mean
        return out

    def log_prob(self, features, targets):
        torch = _torch()
        X = np.asarray(features, dtype=np.float32).reshape(len(features), -1)
        Y = np.asarray(targets, dtype=np.float32).reshape(len(targets), -1)
        Xs = torch.tensor((X - self._x_mean) / self._x_std)
        Ys = torch.tensor((Y - self._y_mean) / self._y_std)
        with torch.no_grad():
            lp = self.flow.log_prob(Ys, Xs).numpy()
        # change of variables for the target standardisation
        return lp - np.log(self._y_std).sum()

    @staticmethod
    def components_to_anisotropy(comp):
        """Map five independent components to a symmetric traceless 3x3 batch.

        comp[...,:] = [b_xx, b_yy, b_xy, b_xz, b_yz]; b_zz = -(b_xx + b_yy).
        """
        comp = np.asarray(comp)
        b = np.zeros(comp.shape[:-1] + (3, 3))
        bxx, byy, bxy, bxz, byz = (comp[..., 0], comp[..., 1], comp[..., 2],
                                   comp[..., 3], comp[..., 4])
        b[..., 0, 0] = bxx
        b[..., 1, 1] = byy
        b[..., 2, 2] = -(bxx + byy)
        b[..., 0, 1] = b[..., 1, 0] = bxy
        b[..., 0, 2] = b[..., 2, 0] = bxz
        b[..., 1, 2] = b[..., 2, 1] = byz
        return b

    def sample_realizable_anisotropy(self, features, base_anisotropy=None, n_per=1):
        """Sample discrepancy components, add an optional base anisotropy, and
        project every draw into the realizable barycentric set.

        Returns the projected anisotropy batch (N, n_per, 3, 3); every entry is
        realizable by construction.
        """
        comp = self.sample(features, n_per=n_per)             # (N, n_per, 5)
        b = self.components_to_anisotropy(comp)               # (N, n_per, 3, 3)
        if base_anisotropy is not None:
            b = b + base_anisotropy[:, None, :, :]
        bp, _ = rz.project_anisotropy(b)
        return bp

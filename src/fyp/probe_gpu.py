"""Batched GPU logistic-regression probe, solved to convergence.

Objective = sklearn's `LogisticRegression(C=1)` objective, so the optimum is the same:
    sum_i CE_i(x_i, y_i)  +  0.5 * ||W||_F^2          (intercept b not penalised)
multinomial softmax over the K classes present in the training set (K > 2), and the single-logit
sigmoid form for K == 2 (sklearn's binary case; it has one coefficient vector, so the penalty differs
from a 2-row softmax). The problem is strictly convex, so the solution does not depend on the optimiser
(unlike sklearn's `max_iter = 100` iterate), which is why this probe is solved to a gradient tolerance
rather than for a fixed iteration count.

Many problems are solved at once: one training set X (n, d), a stack of B keep-masks (B, d). A masked
coordinate is deleted: its weight row is held at exactly 0, which is identical to dropping the column
(L2 drives the weight of an all-zero feature to 0; tested). Optimiser: batched L-BFGS in float64 with a
per-problem Armijo backtracking line search; each problem stops on its own gradient max-norm.
Warm starts (`W0`, `b0`) make single-dimension deletions cheap.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

HISTORY = 10
MAX_ITER = 500
GRAD_TOL = 1e-6          # per-problem max |gradient| of the sum-loss objective (sweeps use 1e-4: 0 prediction flips vs 1e-6 on banking77 dev)
MAX_BACKTRACK = 40


@dataclass
class ProbeState:
    W: torch.Tensor          # (B, d, Kp)   Kp = 1 for binary, else K
    b: torch.Tensor          # (B, Kp)
    n_iter: np.ndarray       # (B,)
    converged: np.ndarray    # (B,) bool
    classes: np.ndarray      # original labels of the K seen classes (index = internal class id)


class LogRegGPU:
    def __init__(self, x_train, y_train, device: str = "cuda"):
        self.device = device
        y_train = np.asarray(y_train)
        self.classes = np.unique(y_train)
        self.K = len(self.classes)
        assert self.K >= 2, "need at least two classes"
        self.Kp = 1 if self.K == 2 else self.K
        self.y_int = np.searchsorted(self.classes, y_train)
        X = torch.as_tensor(np.asarray(x_train), dtype=torch.float64, device=device)
        # Solve on training-mean-centred features: the intercept absorbs the shift (b = b_c - mu @ W), so the
        # optimum is identical, but the problem is far better conditioned (raw embeddings are not centred).
        self.mu = X.mean(0)
        self.X = X - self.mu
        self.y = torch.as_tensor(self.y_int, dtype=torch.long, device=device)
        self.n, self.d = self.X.shape
        self.P = self.d * self.Kp + self.Kp
        if self.Kp == 1:
            self.Y1 = self.y.to(torch.float64)[None, :]                    # (1, n)
        else:
            self.Y1 = torch.nn.functional.one_hot(self.y, self.K).to(torch.float64)[None]   # (1, n, K)
        # initial inverse-Hessian scale: 1 / (L + 1), L = ||X||_2^2 / 4 bounds the data Hessian
        self.gamma0 = float(1.0 / (0.25 * torch.linalg.matrix_norm(self.X, ord=2) ** 2 + 1.0))

    # ---------------------------------------------------------------- objective
    def _split(self, theta, B):
        W = theta[:, : self.d * self.Kp].reshape(B, self.d, self.Kp)
        b = theta[:, self.d * self.Kp:]
        return W, b

    def _fg(self, theta, M3):
        B = theta.shape[0]
        W, b = self._split(theta, B)
        Wm = W * M3
        z = torch.matmul(self.X, Wm) + b[:, None, :]                         # (B, n, Kp)
        if self.Kp == 1:
            z1 = z[..., 0]
            f = (torch.nn.functional.softplus(z1) - self.Y1 * z1).sum(1)
            R = (torch.sigmoid(z1) - self.Y1)[..., None]
        else:
            lse = torch.logsumexp(z, dim=-1)
            f = (lse - z.gather(-1, self.y[None, :, None].expand(B, -1, 1))[..., 0]).sum(1)
            R = torch.softmax(z, dim=-1) - self.Y1
        f = f + 0.5 * (Wm ** 2).sum((1, 2))
        gW = torch.matmul(self.X.T, R) * M3 + Wm
        gb = R.sum(1)
        return f, torch.cat([gW.reshape(B, -1), gb], dim=1)

    # ---------------------------------------------------------------- solver
    def fit(self, masks=None, warm: Optional[ProbeState] = None, max_iter: int = MAX_ITER,
            tol: float = GRAD_TOL) -> ProbeState:
        """masks: (B, d) bool/0-1, True = kept (None = a single all-kept problem). warm: a ProbeState with
        B == len(masks) (or B == 1, broadcast) used as the starting point."""
        dev = self.device
        if masks is None:
            masks = np.ones((1, self.d), dtype=bool)
        M = torch.as_tensor(np.asarray(masks), dtype=torch.float64, device=dev)
        B = M.shape[0]
        M3 = M[:, :, None]
        theta = torch.zeros(B, self.P, dtype=torch.float64, device=dev)
        if warm is not None:
            W0 = warm.W.expand(B, -1, -1) if warm.W.shape[0] == 1 else warm.W
            b0 = warm.b.expand(B, -1) if warm.b.shape[0] == 1 else warm.b
            Wm0 = W0 * M3
            theta = torch.cat([Wm0.reshape(B, -1), b0 + torch.einsum("d,bdk->bk", self.mu, Wm0)], dim=1).clone()
        f, g = self._fg(theta, M3)
        S = torch.zeros(HISTORY, B, self.P, dtype=torch.float64, device=dev)
        Yh = torch.zeros_like(S)
        rho = torch.zeros(HISTORY, B, dtype=torch.float64, device=dev)
        gamma = torch.full((B,), self.gamma0, dtype=torch.float64, device=dev)
        n_hist = 0
        n_iter = torch.zeros(B, dtype=torch.long, device=dev)
        active = g.abs().amax(1) >= tol
        for it in range(max_iter):
            if not bool(active.any()):
                break
            # two-loop recursion (batched; unused history slots have rho = 0 and contribute nothing)
            q = g.clone()
            alphas = []
            for j in range(n_hist - 1, max(-1, n_hist - 1 - HISTORY), -1):
                slot = j % HISTORY
                a = rho[slot] * (S[slot] * q).sum(1)
                q = q - a[:, None] * Yh[slot]
                alphas.append((slot, a))
            r = gamma[:, None] * q
            for slot, a in reversed(alphas):
                beta = rho[slot] * (Yh[slot] * r).sum(1)
                r = r + (a - beta)[:, None] * S[slot]
            d = -r
            gd = (g * d).sum(1)
            bad = gd >= 0
            if bool(bad.any()):                       # not a descent direction: steepest descent for those
                d = torch.where(bad[:, None], -g * gamma[:, None], d)
                gd = (g * d).sum(1)
            # Armijo backtracking, per problem; only problems still without an accepted step are re-evaluated
            t = torch.ones(B, dtype=torch.float64, device=dev)
            ok = ~active
            f_new, g_new, th_new = f.clone(), g.clone(), theta.clone()
            pending = active.nonzero().squeeze(1)
            for _ in range(MAX_BACKTRACK):
                if pending.numel() == 0:
                    break
                th_try = theta[pending] + t[pending, None] * d[pending]
                f_try, g_try = self._fg(th_try, M3[pending])
                good = (f_try <= f[pending] + 1e-4 * t[pending] * gd[pending]) & torch.isfinite(f_try)
                gi = pending[good]
                f_new[gi], g_new[gi], th_new[gi] = f_try[good], g_try[good], th_try[good]
                ok[gi] = True
                pending = pending[~good]
                t[pending] = t[pending] * 0.5
            failed = active & ~ok                      # no acceptable step: treat as converged-by-stall
            s = th_new - theta
            y = g_new - g
            sy = (s * y).sum(1)
            upd = ok & active & (sy > 1e-12 * (s * s).sum(1).clamp_min(1e-300))
            slot = n_hist % HISTORY
            S[slot] = torch.where(upd[:, None], s, torch.zeros_like(s))
            Yh[slot] = torch.where(upd[:, None], y, torch.zeros_like(y))
            rho[slot] = torch.where(upd, 1.0 / sy.clamp_min(1e-300), torch.zeros_like(sy))
            gamma = torch.where(upd, sy / (y * y).sum(1).clamp_min(1e-300), gamma)
            n_hist += 1
            theta, f, g = th_new, f_new, g_new
            n_iter = n_iter + active.long()
            active = active & ~failed & (g.abs().amax(1) >= tol)
        W, b = self._split(theta, B)
        Wm = W * M3
        converged = (g.abs().amax(1) < tol).cpu().numpy()
        return ProbeState(W=Wm.clone(), b=(b - torch.einsum("d,bdk->bk", self.mu, Wm)).clone(), n_iter=n_iter.cpu().numpy(), converged=converged,
                          classes=self.classes)

    # ---------------------------------------------------------------- prediction
    def logits(self, state: ProbeState, x_eval) -> torch.Tensor:
        Xe = torch.as_tensor(np.asarray(x_eval), dtype=torch.float64, device=self.device) if not torch.is_tensor(x_eval) else x_eval
        return torch.matmul(Xe, state.W) + state.b[:, None, :]

    def predict(self, state: ProbeState, x_eval) -> torch.Tensor:
        """(B, m) predicted ORIGINAL labels."""
        z = self.logits(state, x_eval)
        idx = (z[..., 0] > 0).long() if self.Kp == 1 else z.argmax(-1)
        return torch.as_tensor(self.classes, device=self.device)[idx]

    def correct(self, state: ProbeState, x_eval, y_eval, chunk: int = 0) -> torch.Tensor:
        """(B, m) bool: prediction == label. Chunked over B to bound the (B, m, Kp) logits."""
        ye = torch.as_tensor(np.asarray(y_eval), device=self.device)
        B = state.W.shape[0]
        m = len(ye)
        chunk = chunk or max(1, int(2e8 // (8 * m * self.Kp)))
        out = []
        for s in range(0, B, chunk):
            sub = ProbeState(state.W[s:s + chunk], state.b[s:s + chunk], state.n_iter[s:s + chunk],
                             state.converged[s:s + chunk], state.classes)
            out.append(self.predict(sub, x_eval) == ye[None, :])
        return torch.cat(out, 0)

"""Exercise 1 (d), GP Part 2 -- GP regression from scratch (SE kernel, l = 1, n = 2).

Reproduces the lecture's R output in Python (NumPy/SciPy/matplotlib):
  * tune the noise variance sigma^2 by the marginal likelihood
    (solve d/d(sigma^2) log p(y) = 0, i.e. 0.5 * tr(Ky^-1 y y^T Ky^-1 - Ky^-1) = 0),
  * the posterior predictive (part (c) formula) -- shown as a mean +/- 2 sd band.
Run with the py312 env:
  /Users/yaweili/miniforge3/envs/py312/bin/python ex_gp_2_d.py
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(190)  # seed chosen for a clear, well-separated 2-point illustration
n = 2
sigma = 1.0  # true noise sd used to simulate the data


def kern(x, y=None):                       # SE kernel, l = 1
    d = np.subtract.outer(np.ravel(x), np.ravel(x if y is None else y))
    return np.exp(-0.5 * d**2)             # k(x, y) = exp(-0.5 (x - y)^2)


# --- simulate noisy targets  y ~ N(0, K + sigma^2 I) ---
x = rng.standard_normal(n)
Ky = kern(x) + sigma**2 * np.eye(n)
y = rng.multivariate_normal(np.zeros(n), Ky).reshape(-1, 1)
print("x =", np.round(x, 4))
print("y =", np.round(y.ravel(), 4))


# --- tune sigma^2 by the marginal likelihood: 0.5 tr(Kyi y y^T Kyi - Kyi) = 0 ---
def mll_deriv(s2):
    Kyi = np.linalg.inv(kern(x) + s2 * np.eye(n))
    return 0.5 * np.trace(Kyi @ y @ y.T @ Kyi - Kyi)


grid = np.linspace(1e-3, 20, 400)
dvals = np.array([mll_deriv(s) for s in grid])
sc = np.where(np.diff(np.sign(dvals)) != 0)[0]   # sign-change bracket
if len(sc):
    s2 = brentq(mll_deriv, grid[sc[0]], grid[sc[0] + 1])
else:
    s2 = grid[np.argmin(np.abs(dvals))]
print("best sigma^2 =", round(float(s2), 4))


# --- posterior predictive over a grid of test points x* (part (c), zero mean) ---
Kyi = np.linalg.inv(kern(x) + s2 * np.eye(n))
xs = np.linspace(x.min() - 3, x.max() + 3, 200)
Ks = kern(x, xs)                                 # (n, m)
mean = (Ks.T @ Kyi @ y).ravel()                  # predictive mean
var = 1.0 + s2 - np.einsum("aj,ab,bj->j", Ks, Kyi, Ks)  # observable y*: +s2
sd = np.sqrt(np.clip(var, 0.0, None))


# --- figure: (left) sigma^2 tuning, (right) posterior predictive band ---
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.4))

axL.plot(grid, dvals, color="#1f4e79", lw=1.6)
axL.axhline(0, ls="--", color="0.5", lw=1.0)
axL.axvline(s2, ls="--", color="#c0392b", lw=1.2)
axL.set_xlabel(r"$\sigma^2$")
axL.set_ylabel("marginal-likelihood derivative")
axL.set_title(rf"$\sigma^2$ tuning  (root $\approx {s2:.2f}$)")

axR.fill_between(xs, mean - 2 * sd, mean + 2 * sd,
                 color="#1f4e79", alpha=0.18, label=r"$\pm 2\,$sd")
axR.plot(xs, mean, color="#1f4e79", lw=1.8, label="post. mean")
axR.scatter(x, y.ravel(), color="#c0392b", zorder=5, label="data")
axR.set_xlabel(r"$x_*$")
axR.set_ylabel(r"$y_*$")
axR.set_title(r"GP posterior predictive ($n=2$, SE, $\ell=1$)")
axR.legend(loc="upper right", fontsize=8, frameon=False)

fig.tight_layout()
out = "figures/gp_posterior_predictive.pdf"
fig.savefig(out, bbox_inches="tight")
print("saved", out)

"""Exercise 1 (Regularization Part 2): Python port of the R reference solution.

Covers the two sub-questions that produce figures:
  (a)(vi) soft-thresholding operator vs. OLS
  (d)     projected-orthonormal Lasso vs. coordinate-descent Lasso (RMSE boxplot)

Run inside the miniforge `py312` env. Saves all figures to ./figure_ex7/.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figure_ex7"
os.makedirs(OUT, exist_ok=True)

# =============================================================== (a)(vi) =====
# Soft-thresholding operator theta*(rho) for lambda = z = 1, compared to OLS.
rhos = np.arange(-5.0, 5.0 + 1e-9, 0.1)
lam, z = 1.0, 1.0
theta_star = np.where(rhos < -lam, (rhos + lam) / z,
                      np.where(rhos > lam, (rhos - lam) / z, 0.0))
theta_ols = rhos / z

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.plot(rhos, theta_ols, color="tab:red", lw=2, label="OLS")
ax.plot(rhos, theta_star, color="tab:cyan", lw=2, label="soft thresholding")
ax.axhline(0, color="0.8", lw=0.8)
ax.set_xlabel(r"$\rho_j$")
ax.set_ylabel(r"$\theta_j^*$")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/soft_threshold.png", dpi=150)
plt.close(fig)

# ================================================================== (d) =======
def proj_orth_lasso(X, y, lam):
    """Project X onto orthonormal columns via A = V D^{-1/2}, soft-threshold, map back."""
    vals, vecs = np.linalg.eigh(X.T @ X)        # X'X = V D V'
    A = vecs @ np.diag(vals ** -0.5)            # A = V D^{-1/2}  => (XA)'(XA) = I
    X_tilde = X @ A
    proj_theta_ols = X_tilde.T @ y              # OLS for orthonormal design
    proj_theta_star = np.sign(proj_theta_ols) * np.maximum(
        np.abs(proj_theta_ols) - lam, 0.0)      # analytical Lasso solution
    return A @ proj_theta_star                  # back to original coordinates


def lasso(X, y, lam, N):
    """Coordinate descent: cycle through coordinates, apply soft-threshold update."""
    p = X.shape[1]
    theta = np.ones(p)
    for i in range(1, N + 1):
        j = i % p                               # cycle the active coordinate
        mask = np.ones(p)
        mask[j] = 0.0
        rho_j = X[:, j] @ (y - X @ (theta * mask))
        z_j = np.sum(X[:, j] ** 2)
        if rho_j < -lam:
            theta[j] = (rho_j + lam) / z_j
        elif rho_j > lam:
            theta[j] = (rho_j - lam) / z_j
        else:
            theta[j] = 0.0
    return theta


rng = np.random.default_rng(2)
p, n, num_opt_steps = 10, 100, 400
sigma_noise, sigma_signal, lam = 0.1, 1.0, 1.0

rmse_proj, rmse_reg = [], []
for _ in range(100):
    X = rng.normal(scale=sigma_signal, size=(n, p))
    theta_true = rng.normal(size=p)
    theta_true[rng.binomial(1, 0.7, size=p) == 1] = 0.0   # ~70% sparse
    y = X @ theta_true + rng.normal(scale=sigma_noise, size=n)
    rmse_proj.append(n / (n - 1) * np.std(proj_orth_lasso(X, y, lam) - theta_true, ddof=1))
    rmse_reg.append(n / (n - 1) * np.std(lasso(X, y, lam, num_opt_steps) - theta_true, ddof=1))

fig, ax = plt.subplots(figsize=(5.2, 4.4))
bp = ax.boxplot([rmse_proj, rmse_reg], tick_labels=["yes", "no"], patch_artist=True)
for patch, color in zip(bp["boxes"], ["salmon", "tab:cyan"]):
    patch.set_facecolor(color)
for med in bp["medians"]:
    med.set_color("black")
ax.set_xlabel("projected")
ax.set_ylabel(r"$\sqrt{\sum_j (\hat{\theta}_j - \theta_{j,\mathrm{true}})^2 / p}$")
fig.tight_layout()
fig.savefig(f"{OUT}/proj_vs_lasso.png", dpi=150)
plt.close(fig)

print(f"median RMSE  projected={np.median(rmse_proj):.4f}  regular={np.median(rmse_reg):.4f}")
print("figures written to", os.path.abspath(OUT))

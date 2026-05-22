"""Exercise 2(b): regularization demonstrations (Python port of the R solution).

Run inside the miniforge `py312` env. Saves all figures to ./figures/.
"""
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, lasso_path, enet_path
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- (a) data ---
rng = np.random.default_rng(42)
n, p_add = 100, 100
X = rng.standard_normal((n, p_add + 1))
Y = np.sin(X[:, 0]) + rng.normal(scale=0.5, size=n)

x1 = X[:, 0]
order = np.argsort(x1)
xs = x1[order]


def poly_fit(x, y, degree, x_eval):
    """Least-squares polynomial fit of `x`->`y`, predicted at sorted `x_eval`."""
    pf = PolynomialFeatures(degree, include_bias=False)
    model = LinearRegression().fit(pf.fit_transform(x[:, None]), y)
    return model.predict(pf.transform(x_eval[:, None]))


# ----------------------------------------------- under- and over-fitting -----
fig, axes = plt.subplots(2, 1, figsize=(5.0, 7.0))
for ax, deg, title in [(axes[0], 1, "Underfitting (degree 1)"),
                       (axes[1], 7, "Overfitting (degree 7)")]:
    ax.scatter(x1, Y, s=16, color="0.55")
    ax.plot(xs, np.sin(xs), color="tab:blue", lw=2, label="truth")
    ax.plot(xs, poly_fit(x1, Y, deg, xs), color="tab:red", lw=2, label="fit")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fit.png", dpi=150)
plt.close(fig)

# ------------------------------------- L1 / L2 / elastic-net coef paths -------
Xs = StandardScaler().fit_transform(X)
Yc = Y - Y.mean()
fig, axes = plt.subplots(3, 1, figsize=(5.2, 8.4), sharey=True)

a_l1, c_l1, _ = lasso_path(Xs, Yc)
axes[0].plot(np.log(a_l1), c_l1.T, lw=0.8)
axes[0].set_title("$L1$ (lasso)")

alphas_l2 = np.logspace(-2, 3, 100)
c_l2 = np.array([Ridge(alpha=a).fit(Xs, Yc).coef_ for a in alphas_l2])
axes[1].plot(np.log(alphas_l2), c_l2, lw=0.8)
axes[1].set_title("$L2$ (ridge)")

a_en, c_en, _ = enet_path(Xs, Yc, l1_ratio=0.3)
axes[2].plot(np.log(a_en), c_en.T, lw=0.8)
axes[2].set_title("elastic net ($\\alpha=0.3$)")

for ax in axes:
    ax.set_xlabel("$\\log\\lambda$")
    ax.set_ylabel("coefficients")
fig.tight_layout()
fig.savefig(f"{OUT}/paths.png", dpi=150)
plt.close(fig)

# ------------------------------------------- underdetermined problem ----------
XtX = X.T @ X
print("X'X shape :", XtX.shape)
print("rank      :", np.linalg.matrix_rank(XtX), "(< 101 -> singular)")
print("cond. num :", f"{np.linalg.cond(XtX):.3e}")
try:
    beta = np.linalg.solve(XtX, X.T @ Y)
    print("solve returned a vector, but it is meaningless (system singular).")
except np.linalg.LinAlgError as exc:
    print("np.linalg.solve failed:", exc)

# ----------------------------------------------- bias-variance trade-off ------
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(x1, Y, s=14, color=(0, 0, 0, 0.2))
ax.plot(xs, np.sin(xs), color="tab:blue", lw=2.5, label="truth")
colors = ["red", "magenta", "orange", "purple", "green", "brown"]
for deg, c in zip(range(1, 7), colors):
    ax.plot(xs, poly_fit(x1, Y, deg, xs), color=c, lw=1.2, label=f"degree {deg}")
ax.set_xlabel("$x_1$")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(f"{OUT}/bias_variance.png", dpi=150)
plt.close(fig)

# --------------------------------- early stopping (simple neural net) ---------
Xtr, Xval, ytr, yval = train_test_split(X, Y, test_size=0.2, random_state=42)
net = MLPRegressor(hidden_layer_sizes=(50, 50), activation="relu",
                   solver="adam", learning_rate_init=1e-3,
                   warm_start=True, max_iter=1, random_state=42)

train_loss, val_loss = [], []
best, patience, wait, best_epoch = np.inf, 50, 0, 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # silence per-epoch ConvergenceWarning
    for epoch in range(300):
        net.fit(Xtr, ytr)            # one epoch (max_iter=1 + warm_start)
        train_loss.append(mean_squared_error(ytr, net.predict(Xtr)))
        v = mean_squared_error(yval, net.predict(Xval))
        val_loss.append(v)
        if v < best - 1e-4:
            best, best_epoch, wait = v, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                break

fig, ax = plt.subplots(figsize=(5.4, 4.0))
ax.plot(train_loss, color="tab:blue", label="training loss")
ax.plot(val_loss, color="tab:red", label="validation loss")
ax.axvline(best_epoch, color="0.4", ls="--", lw=1,
           label=f"best epoch = {best_epoch}")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.set_title("Early stopping (50 epochs patience)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/early_stopping.png", dpi=150)
plt.close(fig)

print("stopped at epoch", epoch, "| best epoch", best_epoch)
print("figures written to", os.path.abspath(OUT))

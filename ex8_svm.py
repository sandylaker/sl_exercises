"""Exercise 2 (SVM Part 1): stochastic subgradient descent for the soft-margin
SVM in the primal formulation (PEGASOS), Python port of sol_linsvm_subgradient.

Generates figures/svm_pegasos.png and prints a comparison against sklearn's SVC.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def pegasos_linear(y, X, nr_iter=50000, lam=1.0, alpha=0.01, seed=0):
    """Primal soft-margin SVM via stochastic subgradient descent (PEGASOS).

    X: design matrix with a leading column of 1s (intercept); y in {-1, +1}.
    Returns theta = (theta_0, theta_1, ...).
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    theta = rng.standard_normal(p)
    for _ in range(nr_iter):
        i = rng.integers(n)
        f_i = X[i] @ theta                 # margin score with current theta
        theta = (1 - lam * alpha) * theta  # weight decay (regularization)
        if y[i] * f_i < 1:                 # inside the margin -> subgradient step
            theta = theta + alpha * y[i] * X[i]
    return theta


# ---- toy data: two Gaussians (cf. mlbench.twonorm) ------------------------
rng = np.random.default_rng(2)
n, C = 100, 1.0
a = 2 / np.sqrt(2)
Xpos = rng.standard_normal((n // 2, 2)) + a
Xneg = rng.standard_normal((n // 2, 2)) - a
X = np.vstack([Xpos, Xneg])
y = np.r_[np.ones(n // 2), -np.ones(n // 2)]

# ---- our PEGASOS solution -------------------------------------------------
Xb = np.c_[np.ones(n), X]                  # prepend intercept column
theta = pegasos_linear(y, Xb, lam=1 / (C * n))

# ---- compare to sklearn linear SVM (no scaling) ---------------------------
svc = SVC(kernel="linear", C=C).fit(X, y)
w_svc = np.r_[svc.intercept_, svc.coef_[0]]

f_peg = Xb @ theta
print("pegasos theta :", np.round(theta, 3))
print("sklearn theta :", np.round(w_svc, 3))
print("pegasos  misclassified:", int(np.sum(np.sign(f_peg) != y)))
print("sklearn  misclassified:", int(np.sum(svc.predict(X) != y)))

# ---- plot data + both decision boundaries ---------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(*X[y == 1].T, marker="+", c="#0072B2", label="class +1")
ax.scatter(*X[y == -1].T, marker="_", c="#D55E00", label="class -1")
xs = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
ax.plot(xs, -(theta[0] + theta[1] * xs) / theta[2], "-",
        c="#009E73", lw=2, label="PEGASOS")
ax.plot(xs, -(w_svc[0] + w_svc[1] * xs) / w_svc[2], "--",
        c="black", lw=1.5, label="sklearn SVC")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "svm_pegasos.png"), dpi=150)
print("saved", os.path.join(OUT, "svm_pegasos.png"))

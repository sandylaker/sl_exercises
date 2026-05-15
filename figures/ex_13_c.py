import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Grids over (p, n); B = MC sample size per cell
p_seq = np.linspace(0.01, 0.99, 100)
n_seq = np.arange(10, 510, 100)
rng = np.random.default_rng(0)

# Approx KLD by Monte Carlo: E_f[log f - log q] over B draws
def kld_approx(n, p, B=10000):
    x = rng.binomial(n, p, size=B)
    lf = stats.binom.logpmf(x, n, p)
    lq = stats.norm.logpdf(x, n*p, np.sqrt(n*p*(1-p)))
    return max(np.mean(lf - lq), 0)  # clamp tiny negatives

# Evaluate KLD at each (n, p) grid point
Z = np.array([[kld_approx(n_, p_) for n_ in n_seq]
              for p_ in p_seq])

# Custom blue->red colormap (matches the R original)
cmap = LinearSegmentedColormap.from_list(
    'c', ['lightblue', 'blue', 'red', 'darkred'])

# Filled-contour heatmap of the KLD over (p, n)
fig, ax = plt.subplots(figsize=(7, 4))
cs = ax.contourf(p_seq, n_seq, Z.T, levels=50, cmap=cmap)
fig.colorbar(cs, ax=ax, label='KLD')
ax.set_xlabel('p'); ax.set_ylabel('n')
plt.savefig('ex_13_c.pdf', bbox_inches='tight')

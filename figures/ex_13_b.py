import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Setup: Bin(n=100, p=0.5), draw B=1000 samples
n, p, B = 100, 0.5, 1000
X = stats.binom.rvs(n, p, size=B, random_state=0)

# Optimal Gaussian moments (match the binomial)
mu0, sd0 = n*p, np.sqrt(n*p*(1-p))
xs = np.linspace(10, 100, 500)

# 4 candidate Gaussians: (mean, sd, color, label)
pdfs = [(mu0,      sd0,     'green',  'optimal'),
        (mu0 - 10, sd0,     'blue',   'shift'),
        (mu0,      2*sd0,   'orange', 'scale+'),
        (mu0 + 20, p*(1-p), 'red',    'narrow')]

# Histogram of samples, overlay each candidate density
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(X, bins=25, range=(10, 100), density=True,
        color='lightgray', edgecolor='black')
for m, s, c, lbl in pdfs:
    ax.plot(xs, stats.norm.pdf(xs, m, s), color=c, label=lbl)
ax.set_xlabel('x'); ax.set_ylabel('density'); ax.legend()
plt.savefig('ex_13_b.pdf', bbox_inches='tight')

# KLD (up to additive constant) from the closed form
def kld(mu, s2):
    return 0.5*np.log(s2) + 0.5/s2 * (n*p*(1-p) + (n*p - mu)**2)

for m, s, _, lbl in pdfs:
    print(f'{lbl:8s}: {kld(m, s**2):.4f}')

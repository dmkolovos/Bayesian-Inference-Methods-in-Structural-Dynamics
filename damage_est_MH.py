# damage_est_MH.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
from solver import node_disp_from_d

# ---------------- SETTINGS ----------------
MEAS_CSV   = "measurement_node3.csv"
NOISE_REL  = 0.02
A, B       = 0.0, 1.0
N_SAMPLES  = 50000
RNG        = np.random.default_rng(123)

# ---------------- LIKELIHOOD ----------------
def gaussian_loglikelihood(u_m: np.ndarray, u_model: np.ndarray, sigma_vec: np.ndarray) -> float:
    """log[p(u|d)]"""
    return -0.5 * np.sum(((u_m - u_model) / sigma_vec) ** 2
                         + np.log(2.0 * np.pi) + 2.0 * np.log(sigma_vec))

def log_prior(d: np.ndarray) -> float:
    """Uniform prior"""
    if np.any((d < A) | (d > B)):
        return -np.inf
    return 0.0

def log_posterior(d: np.ndarray, u_m: np.ndarray, sigma_vec: np.ndarray) -> float:
    """log[p(d|u)] ∝ log[p(u|d)] + log[p(d)]"""
    u_model = node_disp_from_d(float(d[0]), float(d[1]), float(d[2]))
    return gaussian_loglikelihood(u_m, u_model, sigma_vec) + log_prior(d)

# ---------------- METROPOLIS-HASTINGS ----------------
def mh_uniform(u_meas: np.ndarray, sigma_vec: np.ndarray, n_steps: int = N_SAMPLES,
               rng: np.random.Generator = RNG) -> np.ndarray:
    d_curr = rng.uniform(A, B, size=3)
    f_curr = log_posterior(d_curr, u_meas, sigma_vec)
    chain, accepts = [], 0

    for t in range(n_steps):
        d_prop = rng.uniform(A, B, size=3)
        f_prop = log_posterior(d_prop, u_meas, sigma_vec)

        if np.log(rng.uniform(0, 1)) < (f_prop - f_curr):
            d_curr, f_curr = d_prop, f_prop
            accepts += 1
        chain.append(d_curr.copy())

    acc_rate = accepts / n_steps
    print(f"MH (Uniform proposal) acceptance rate: {acc_rate:.3f}")
    return np.array(chain)

def mh_gaussian(u_meas: np.ndarray, sigma_vec: np.ndarray, n_steps: int = N_SAMPLES,
                rng: np.random.Generator = RNG, tau: float = 0.05) -> np.ndarray:
    d_curr = rng.uniform(A, B, size=3)
    f_curr = log_posterior(d_curr, u_meas, sigma_vec)
    chain, accepts = [], 0

    for t in range(n_steps):
        d_prop = d_curr + tau * rng.normal(size=3)
        f_prop = log_posterior(d_prop, u_meas, sigma_vec)

        if np.log(rng.uniform(0, 1)) < (f_prop - f_curr):
            d_curr, f_curr = d_prop, f_prop
            accepts += 1
        chain.append(d_curr.copy())

    acc_rate = accepts / n_steps
    print(f"MH (Gaussian proposal) acceptance rate: {acc_rate:.3f}")
    return np.array(chain)

# ---------------- PLOTS ----------------
def plot_posterior_histograms(samples: np.ndarray, label: str, bins: int = 40, color: str = "steelblue"):
    """Create and save histogram + KDE plots for each parameter"""
    n_params = samples.shape[1]

    for i in range(n_params):
        s = samples[:, i]
        plt.figure(figsize=(8, 6))
        plt.hist(s, bins=bins, density=True, alpha=0.6, color=color, label="Histogram")
        if len(s) > 2:
            kde = gaussian_kde(s)
            x = np.linspace(s.min(), s.max(), 500)
            plt.plot(x, kde(x), 'k--', lw=2, label="KDE")
        plt.xlabel(fr"$d_{i+1}$")
        plt.ylabel("Density")
        plt.title(fr"Posterior of $d_{i+1}$ ({label})")
        plt.grid(True, alpha=0.2)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(f"{label.replace(' ', '_')}_hist_d{i+1}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

# ---------------- MAIN ----------------
def main():
    u_meas = np.loadtxt(MEAS_CSV, delimiter=",", skiprows=1)
    sigma_vec = NOISE_REL * np.abs(u_meas)

    print("u_meas =", u_meas)
    print("sigma_vec =", sigma_vec)

    # ---- MH Uniform ----
    samples_uniform = mh_uniform(u_meas, sigma_vec)
    np.savetxt("posterior_samples_mh_uniform.csv", samples_uniform, delimiter=",",
               header="d1,d2,d3", comments="")
    print("Saved: posterior_samples_mh_uniform.csv")
    print("Posterior mean :", samples_uniform.mean(axis=0))
    print("Posterior std  :", samples_uniform.std(axis=0))
    plot_posterior_histograms(samples_uniform, label="MH Uniform", color="steelblue")

    # ---- MH Gaussian ----
    samples_gaussian = mh_gaussian(u_meas, sigma_vec)
    np.savetxt("posterior_samples_mh_gaussian.csv", samples_gaussian, delimiter=",",
               header="d1,d2,d3", comments="")
    print("Saved: posterior_samples_mh_gaussian.csv")
    print("Posterior mean :", samples_gaussian.mean(axis=0))
    print("Posterior std  :", samples_gaussian.std(axis=0))
    plot_posterior_histograms(samples_gaussian, label="MH Gaussian", color="orange")

if __name__ == "__main__":
    main()
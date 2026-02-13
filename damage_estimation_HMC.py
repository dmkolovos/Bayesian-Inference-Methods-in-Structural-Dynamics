# damage_estimation_HMC.py
import numpy as np
from solver import node_disp_from_d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

MEAS_CSV = "measurement_node3.csv"
A, B = 0.0, 1.0
noise_rel = 0.02
EPS = 1e-7                        # step size for grad
DT = 0.002                        # leapfrog integrator step size
L = 10                            # number of leapfrog integrator steps per iter
N_SAMPLES = 1000
MASS = np.eye(3)                  # mass matrix
RNG = np.random.default_rng(123)

# Log posterior and grad
def log_posterior(d: np.ndarray, u_meas: np.ndarray, sigma_vec: np.ndarray) -> float:
    if np.any((d < A) | (d > B)):
        return -np.inf
    u_model = node_disp_from_d(float(d[0]), float(d[1]), float(d[2]))
    loglike = -0.5 * np.sum(((u_meas - u_model) / sigma_vec)**2 + np.log(2 * np.pi) + 2 * np.log(sigma_vec))
    return loglike

def grad_logpost(d: np.ndarray, u_meas: np.ndarray, sigma_vec: np.ndarray, eps=EPS) -> np.ndarray:
    g = np.zeros_like(d)
    for i in range(len(d)):
        e = np.zeros_like(d)
        e[i] = 1.0
        g[i] = (log_posterior(d + eps*e, u_meas, sigma_vec) -
                log_posterior(d - eps*e, u_meas, sigma_vec)) / (2 * eps)
    return g

# Hamiltonian
def kinetic(p: np.ndarray, M_inv: np.ndarray) -> float:
    return 0.5 * p.T @ M_inv @ p

def leapfrog_integrator(d, p, u_meas, sigma_vec, dt, L, M_inv):
    p = p + 0.5 * dt * grad_logpost(d, u_meas, sigma_vec)
    for _ in range(L):
        d = d + dt * (M_inv @ p)
        if np.any((d < A) | (d > B)):
            d = np.clip(d, A, B)
            p = -p
        g = grad_logpost(d, u_meas, sigma_vec)
        if _ < L - 1:
            p = p + dt * g
    p = p + 0.5 * dt * g
    return d, p

# HMC
def hmc(u_meas, sigma_vec, n_samples=N_SAMPLES, dt=DT, L=L, mass=MASS, rng=RNG):
    D = len(mass)
    M_inv = np.linalg.inv(mass)
    chain = []
    accepts = 0

    # initialization from uniform prior
    d_curr = rng.uniform(A, B, size=D)
    f_curr = log_posterior(d_curr, u_meas, sigma_vec)

    for t in range(n_samples):
        p_curr = rng.multivariate_normal(np.zeros(D), mass)

        d_prop, p_prop = leapfrog_integrator(d_curr.copy(), p_curr.copy(), u_meas, sigma_vec, dt, L, M_inv)
        f_prop = log_posterior(d_prop, u_meas, sigma_vec)

        H_curr = -f_curr + kinetic(p_curr, M_inv)
        H_prop = -f_prop + kinetic(p_prop, M_inv)
        log_alpha = -H_prop + H_curr

        if np.log(rng.random()) < log_alpha:
            d_curr, f_curr = d_prop, f_prop
            accepts += 1

        chain.append(d_curr.copy())

        if (t+1) % 50 == 0:
            print(f"Iter {t+1:4d}: d={d_curr}")

    chain = np.array(chain)
    print(f"Final acceptance: {accepts / n_samples:.3f}")
    return chain

def main():
    u_meas = np.loadtxt(MEAS_CSV, delimiter=",", skiprows=1).ravel()
    sigma_vec = noise_rel * np.abs(u_meas)

    print("u_meas =", u_meas)
    samples = hmc(u_meas, sigma_vec)
    burn = int(0.4 * len(samples))
    post = samples[burn:]
    np.savetxt("posterior_samples_hmc.csv", samples, delimiter=",",
               header="d1,d2,d3", comments="")
    print("Posterior mean:", post.mean(axis=0))
    print("Posterior std :", post.std(axis=0))

    # PLOTS
    labels = [r"$d_1$", r"$d_2$", r"$d_3$"]
    N, D = post.shape

    for j in range(D):
        x = post[:, j]
        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=40, density=True, alpha=0.6, color="steelblue", label="Histogram")

        if len(x) >= 2:
            kde = gaussian_kde(x)
            xs = np.linspace(x.min(), x.max(), 500)
            plt.plot(xs, kde(xs), "k--", lw=2, label="KDE")

        m, s = x.mean(), x.std(ddof=1)
        plt.axvline(m, color="red", lw=1.5, label=f"mean={m:.4f}")
        plt.axvline(m-2*s, color="gray", lw=1, ls="--")
        plt.axvline(m+2*s, color="gray", lw=1, ls="--")

        plt.xlabel(labels[j]); plt.ylabel("Density")
        plt.title(f"Posterior of {labels[j]} (HMC)")
        plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(f"HMC_hist_d{j+1}.png", dpi=150)
        plt.show()

if __name__ == "__main__":
    main()
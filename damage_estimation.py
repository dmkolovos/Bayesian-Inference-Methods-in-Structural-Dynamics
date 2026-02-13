# damage_estimation.py
import numpy as np
from solver import node_disp_from_d
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

measurements_csv = "measurement_node3.csv"
a, b = 0.0, 1.0
noise_rel = 0.02

k0 = 0.20
dk = 0.5
min_k = 1e-6

max_iters = 5000
end_grad = 1e-6
end_impr = 1e-8

dx = 1e-5

def sample_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

def gaussian_loglikelihood(u_m: np.ndarray, u_model: np.ndarray, sigma_vec: np.ndarray) -> float: #log[p(u|d)]
    return -0.5 * np.sum(((u_m - u_model) / sigma_vec) ** 2 + np.log(2.0 * np.pi) + 2.0 * np.log(sigma_vec))

def log_prior(d: np.ndarray) -> float:  #log[p(d)]
    return 0.0

def log_posterior(d: np.ndarray, u_m: np.ndarray, sigma_vec: np.ndarray) -> float:  #log[p(d|u)] - log[p(d)]
    u_model = node_disp_from_d(float(d[0]), float(d[1]), float(d[2]))
    return gaussian_loglikelihood(u_m, u_model, sigma_vec) + log_prior(d)

# Gradient Method
def grad_logpost_cdiff(d: np.ndarray, u_m: np.ndarray, sigma_vec: np.ndarray) -> np.ndarray:  # central diff
    grad = np.zeros(3)
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        d_2  = d + dx * e; f_2  = log_posterior(d_2, u_m, sigma_vec)
        d_1 = d - dx * e; f_1 = log_posterior(d_1, u_m, sigma_vec)
        grad[i] = (f_2 - f_1) / (2.0 * dx)
    return grad

def rejection_sampling(u_m, sigma_vec, N_samples=10000):
    samples = []
    k = len(u_m)

    M = 1.0 / ((2.0 * np.pi) ** (k / 2) * np.prod(sigma_vec))
    print(f"M = {M:.3e}")

    for _ in range(N_samples * 10):
        d = np.random.uniform(0.0, 1.0, size=3)
        u_model = node_disp_from_d(d[0], d[1], d[2])
        f = np.exp(gaussian_loglikelihood(u_m, u_model, sigma_vec))
        U = np.random.uniform(0.0, 1.0)
        if U <= f/M :
            samples.append(d)
        if len(samples) >= N_samples:
            break

    samples = np.array(samples)
    accept_rate = len(samples) / (N_samples * 5)
    print(f"Accepted {len(samples)} / {N_samples * 5} samples ({accept_rate * 100:.2f}%)")
    return samples

def plot_posterior_histograms(samples, bins=30, color='steelblue'):
    n_params = samples.shape[1]
    for i in range(n_params):
        param_samples = samples[:, i]
        plt.figure(figsize=(8, 6))
        plt.hist(param_samples, bins=bins, density=True, alpha=0.6, color=color, label='Histogram')
        kde = gaussian_kde(param_samples)
        x_vals = np.linspace(param_samples.min(), param_samples.max(), 500)
        plt.plot(x_vals, kde(x_vals), 'k--', linewidth=2, label='KDE')
        if i == 0:
            plt.title(r'$\text{Posterior of } d_1$')
            plt.xlabel(r'$d_1$')
        elif i == 1:
            plt.title(r'$\text{Posterior of } d_2$')
            plt.xlabel(r'$d_2$')
        elif i == 2:
            plt.title(r'$\text{Posterior of } d_3$')
            plt.xlabel(r'$d_3$')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    u_m = np.loadtxt(measurements_csv, delimiter=",", skiprows=1)
    sigma_vec = noise_rel * np.abs(u_m)         #approx
    print(sigma_vec)
    d = np.array([sample_uniform(a, b),  #initialization
                  sample_uniform(a, b),
                  sample_uniform(a, b)], dtype=float)
    f = log_posterior(d, u_m, sigma_vec)
    k = k0
    print(f"[initialization] d={d}, log[p(d|u)]={f:.6f}")

    history = []
    for iter in range(1, max_iters + 1):
        grad = grad_logpost_cdiff(d, u_m, sigma_vec)
        grad_norm = np.linalg.norm(grad, ord=2)

        if grad_norm < end_grad:
            print(f"Stop @ iter {iter}: ||grad||={grad_norm:.3e}")
            break

        dir_vec = grad / grad_norm

        improved = False
        while k >= min_k:
            d_try = d + k * dir_vec
            f_try = log_posterior(d_try, u_m, sigma_vec)
            if f_try > f:
                d, f = d_try, f_try
                improved = True
                break
            k *= dk

        history.append([iter, d[0], d[1], d[2], f, grad_norm, k])

        print(f"iter {iter:03d} | log[p(d|u)]={f: .6f} | d={d} | ||grad||={grad_norm:.3e} | k={k:.3e} "
              f"{'ACCEPT' if improved else 'no-improve'}")

        if not improved or k < min_k:
            print("No sufficient improvement / k too small. Stopping.")
            break

        k = k0

    np.savetxt("iterative_history.csv",
               np.asarray(history),
               delimiter=",",
               header="iter,d1,d2,d3,logpost,grad_norm,k",
               comments="")
    print("\n=== RESULT ===")
    print("d =", d)
    print("log[p(d|u)] =", f)

    samples = rejection_sampling(u_m, sigma_vec, N_samples=10000)

    np.savetxt("posterior_samples_rejection.csv", samples, delimiter=",",
               header="d1,d2,d3", comments="")

    plot_posterior_histograms(samples)

if __name__ == "__main__":
    main()
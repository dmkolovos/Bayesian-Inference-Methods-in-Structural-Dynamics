# damage_estimation_TMCMC.py
import numpy as np
from solver import node_disp_from_d
from dataclasses import dataclass

MEAS_CSV    = "measurement_node3.csv"
a, b        = 0.0, 1.0
d_dim       = 3
n_part      = 1000
target_cov  = 1.0
max_iter    = 30
mh_steps    = 15
beta_scale  = 0.6
noise_rel   = 0.02
RNG         = np.random.default_rng(123)

@dataclass
class TMCMCState:
    beta: float
    particles: np.ndarray
    loglike: np.ndarray
    logprior: np.ndarray

def gaussian_loglikelihood(u_meas: np.ndarray, u_model: np.ndarray, sigma_vec: np.ndarray) -> float:
    z = (u_meas - u_model) / sigma_vec
    return -0.5 * np.sum(z**2 + np.log(2.0*np.pi) + 2.0*np.log(sigma_vec))

def log_prior_uniform(d: np.ndarray) -> float:
    return 0.0 if np.all((d >= 0.0) & (d <= 1.0)) else -np.inf

def log_post_beta(d: np.ndarray, beta: float, u_meas: np.ndarray, sigma_vec: np.ndarray) -> float:
    logprior = log_prior_uniform(d)
    u_model = node_disp_from_d(float(d[0]), float(d[1]), float(d[2]))
    loglikelihood = gaussian_loglikelihood(u_meas, u_model, sigma_vec)
    return logprior + beta * loglikelihood

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = weights.size
    positions = (rng.random() + np.arange(N)) / N
    cs = np.cumsum(weights)
    idx = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cs[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx

def choose_next_beta(beta_curr: float, loglike: np.ndarray, target_cov: float = target_cov) -> float:
    if beta_curr >= 1.0:
        return 1.0
    s_min, s_max = 0, 1.0 - beta_curr
    def CoV(s: float) -> float:
        w = np.exp(s * loglike)
        w /= np.sum(w)
        m = np.mean(w)
        cov = np.sqrt(np.mean((w - m)**2)) / m
        return cov
    if CoV(s_max) < target_cov:
        return beta_curr + s_max
    for _ in range(100):
        s_mid = 0.5*(s_min + s_max)
        if CoV(s_mid) < target_cov:
            s_min = s_mid
        else:
            s_max = s_mid
    return beta_curr + s_min

def mh_local_moves(particles: np.ndarray,
                   beta: float,
                   u_meas: np.ndarray,
                   sigma_vec: np.ndarray,
                   cov,
                   steps: int = mh_steps,
                   rng: np.random.Generator = RNG) -> np.ndarray:
    N, D = particles.shape
    out = particles.copy()
    cov_scaled = beta_scale**2 *cov
    for i in range(N):
        d_curr = out[i]
        f_curr = log_post_beta(d_curr, beta, u_meas, sigma_vec)
        for _ in range(steps):
            d_prop = rng.multivariate_normal(d_curr, cov_scaled)
            f_prop = log_post_beta(d_prop, beta, u_meas, sigma_vec)
            log_alpha = f_prop - f_curr
            if np.log(rng.random()) < log_alpha:
                d_curr, f_curr = d_prop, f_prop
        out[i] = d_curr
    return out

def tmcmc(u_meas: np.ndarray,
          sigma_vec: np.ndarray,
          n_part: int = n_part,
          target_cov: float = target_cov,
          max_stage: int = max_iter,
          rng: np.random.Generator = RNG) -> tuple[np.ndarray, list[float]]:
    parts = rng.uniform(a, b, size=(n_part, d_dim))
    loglike = np.zeros(n_part)
    logprior = np.zeros(n_part)
    for i in range(n_part):
        logprior[i] = log_prior_uniform(parts[i])
        u_model = node_disp_from_d(float(parts[i,0]), float(parts[i,1]), float(parts[i,2]))
        loglike[i] = gaussian_loglikelihood(u_meas, u_model, sigma_vec)

    state = TMCMCState(beta=0.0, particles=parts, loglike=loglike, logprior=logprior)
    beta_history = [state.beta]
    print(f"Stage 0: β -> {state.beta:.3f}")

    for stage in range(1, max_stage):
        beta_new = choose_next_beta(state.beta, state.loglike, target_cov=target_cov)
        s = beta_new - state.beta
        print(f"Stage {stage}: β -> {beta_new:.5f}  (Δβ={s:.5f})")

        w = np.exp(s * state.loglike)
        w /= (np.sum(w))

        mean = np.average(state.particles, axis=0, weights=w)
        diff = state.particles - mean
        cov = np.dot(w * diff.T, diff) / np.sum(w)

        idx = systematic_resample(w, rng)
        parts = state.particles[idx]
        loglike = state.loglike[idx]

        parts = mh_local_moves(parts, beta=beta_new, u_meas=u_meas, sigma_vec=sigma_vec,
                               cov = cov, rng=rng)

        for i in range(n_part):
            u_model = node_disp_from_d(float(parts[i,0]), float(parts[i,1]), float(parts[i,2]))
            loglike[i] = gaussian_loglikelihood(u_meas, u_model, sigma_vec)

        state = TMCMCState(beta=beta_new, particles=parts, loglike=loglike, logprior=np.zeros(n_part))
        beta_history.append(beta_new)

        if np.isclose(beta_new, 1.0):
            break

    return state.particles, beta_history

def main():
    u_meas = np.loadtxt(MEAS_CSV, delimiter=",", skiprows=1)

    sigma_vec = noise_rel * np.abs(u_meas)

    print("u_meas   =", u_meas)
    print("sigma_vec=", sigma_vec)

    samples, betas = tmcmc(u_meas, sigma_vec, n_part=n_part, target_cov=target_cov,
                           max_stage=max_iter, rng=RNG)

    np.savetxt("posterior_samples_tmcmc.csv", samples, delimiter=",",
               header="d1,d2,d3", comments="")
    np.savetxt("tmcmc_betas.csv", np.asarray(betas), delimiter=",",
               header="beta", comments="")

    mean = samples.mean(axis=0); std = samples.std(axis=0)
    print("Posterior mean :", mean)

if __name__ == "__main__":
    main()
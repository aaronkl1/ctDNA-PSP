import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm
import os

D2_FIXED     = 0.75
D3_FIXED     = 0.75
SIGMA2_FIXED = 0.03
MP_FIXED     = 1e-6

# ctDNA clearance: half-life ≈ 40 min → 25/day
EPSILON = 25.0

# y0 cap (must match fitting constraint)
Y0_CAP = 2.0

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir    = os.path.join(script_dir, 'output_v3')
os.makedirs(out_dir, exist_ok=True)

# Log-space params: sample log10(param) ~ TruncNormal, then exponentiate.
# Linear params:   sample param ~ TruncNormal directly.
#
# Stats computed from 726 v3-fitted patients.

LOG_PARAMS = {
    # param: (log10_mean, log10_std, log10_lo, log10_hi)
    'mT_tilde':     (-1.636, 1.481, -6.0,   0.0  ),
    's_tilde':      (-4.821, 2.172, -8.0,  -2.0  ),
    'sigma1_tilde': (-0.839, 1.364, -3.0,   1.699),
}

LIN_PARAMS = {
    # param: (mean, std, lo, hi)
    'r':      (0.0181, 0.0266, 0.0001, 0.10),
    'mu':     (0.4947, 0.3057, 0.01,   1.0 ),
    'p':      (0.3426, 0.2670, 0.01,   1.0 ),
    'k':      (0.4313, 0.3559, 0.001,  1.0 ),
    'D1':     (0.7592, 0.1031, 0.5,    1.0 ),
    'x0':     (0.7584, 0.1794, 0.5,    1.0 ),
    'y_frac': (0.3940, 0.3366, 0.0,    1.0 ),
}


def sample_log(param, n):
    mean, std, lo, hi = LOG_PARAMS[param]
    a, b = (lo - mean) / std, (hi - mean) / std
    log_vals = truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
    return 10.0 ** log_vals


def sample_lin(param, n):
    mean, std, lo, hi = LIN_PARAMS[param]
    if std < 1e-12:
        return np.full(n, mean)
    a, b = (lo - mean) / std, (hi - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=n)


def y_inf(p, k, sigma1_tilde, s_tilde):
    """Immune saturation limit as x → ∞ (with fixed D2, D3)."""
    G = D2_FIXED * p - (1.0 - D3_FIXED) * k
    disc = G ** 2 + 4.0 * sigma1_tilde * s_tilde
    return (G + np.sqrt(max(0.0, disc))) / (2.0 * sigma1_tilde)


N_SIMS  = 10000
T_SPAN  = (0, 300)
T_EVAL  = np.arange(0, 301, dtype=float)   # daily, 301 points

np.random.seed(42)

mT_s   = sample_log('mT_tilde',     N_SIMS)
s_s    = sample_log('s_tilde',      N_SIMS)
sig1_s = sample_log('sigma1_tilde', N_SIMS)
r_s    = sample_lin('r',      N_SIMS)
mu_s   = sample_lin('mu',     N_SIMS)
p_s    = sample_lin('p',      N_SIMS)
k_s    = sample_lin('k',      N_SIMS)
D1_s   = sample_lin('D1',     N_SIMS)
x0_s   = sample_lin('x0',     N_SIMS)
yf_s   = sample_lin('y_frac', N_SIMS)

# phi sampled independently for ctDNA only (not in ODE)
phi_s  = np.random.uniform(0.20, 0.90, N_SIMS)

records       = []
param_records = []

for i in range(N_SIMS):
    r            = r_s[i]
    mu           = mu_s[i]
    mT_tilde     = mT_s[i]
    s_tilde      = s_s[i]
    p            = p_s[i]
    sigma1_tilde = sig1_s[i]
    k_ex         = k_s[i]       # exhaustion rate (avoid shadowing built-in k)
    D1           = D1_s[i]
    x0           = x0_s[i]
    y_frac_raw   = yf_s[i]
    phi          = phi_s[i]

    # y0 <= Y0_CAP * y_inf constraint
    yi          = y_inf(p, k_ex, sigma1_tilde, s_tilde)
    max_y0      = Y0_CAP * yi
    max_y_frac  = min(1.0, max_y0 / max(1.0 - x0, 1e-10))
    y_frac      = min(y_frac_raw, max_y_frac)
    y_frac_clamped = y_frac < y_frac_raw

    # initial conditions (V(0) = 1 by construction)
    x_init = x0
    i_tot  = 1.0 - x_init
    y_init = i_tot * y_frac
    z_init = i_tot - y_init

    # ctDNA parameters
    d   = r * phi / (1.0 - phi)
    b   = r + d

    K_0 = D1 * (mu * y_init) / (mT_tilde + x_init)

    d_eff = d + K_0
    r_eff = b - d_eff       # = r - K_0

    denom = max(d, 1e-12)
    alpha = (EPSILON + r) / x_init
    beta  = (mu / denom) * (EPSILON + r) / x_init

    c_init = 1.0   # normalised ctDNA at baseline

    # ODE
    def model(t, state):
        x, y, z, c = state
        x = max(0.0, x); y = max(0.0, y)
        z = max(0.0, z); c = max(0.0, c)

        T_val      = max(1e-12, x)
        kill_denom = mT_tilde + T_val
        mP_denom   = MP_FIXED + T_val

        kill_term   = D1 * mu * y * T_val / kill_denom
        inhib_term  = (MP_FIXED + D2_FIXED * T_val) / mP_denom
        prolif_term = p * T_val * y / kill_denom
        exhaust     = (1.0 - D3_FIXED) * k_ex * T_val * y / mP_denom

        dxdt = r * x - kill_term
        dydt = s_tilde + inhib_term * prolif_term - sigma1_tilde * y**2 - exhaust
        dzdt = exhaust - SIGMA2_FIXED * z
        dcdt = alpha * x + beta * (kill_term / mu) - EPSILON * c

        return [dxdt, dydt, dzdt, dcdt]

    sol = solve_ivp(model, T_SPAN, [x_init, y_init, z_init, c_init],
                    t_eval=T_EVAL, method='Radau',
                    rtol=1e-5, atol=1e-8)

    if sol.success:
        x_r, y_r, z_r, c_r = sol.y
    else:
        x_r = np.full(len(T_EVAL), np.nan)
        y_r = np.full(len(T_EVAL), np.nan)
        z_r = np.full(len(T_EVAL), np.nan)
        c_r = np.full(len(T_EVAL), np.nan)

    v_r = x_r + y_r + z_r

    records.append(pd.DataFrame({
        'Simulation_ID': np.full(len(T_EVAL), i, dtype=int),
        'Time':          T_EVAL,
        'V_scaled':      v_r,
        'X_scaled':      x_r,
        'Y_scaled':      y_r,
        'Z_scaled':      z_r,
        'C_scaled':      c_r,
    }))

    param_records.append({
        'Simulation_ID':  i,
        # free ODE params
        'r':              r,
        'mu':             mu,
        'mT_tilde':       mT_tilde,
        's_tilde':        s_tilde,
        'p':              p,
        'sigma1_tilde':   sigma1_tilde,
        'k':              k_ex,
        'D1':             D1,
        'x0':             x0,
        'y_frac':         y_frac,
        'y_frac_clamped': y_frac_clamped,
        # fixed ODE params
        'D2':             D2_FIXED,
        'D3':             D3_FIXED,
        'sigma2':         SIGMA2_FIXED,
        'mP_tilde':       MP_FIXED,
        # ctDNA-derived
        'phi':            phi,
        'd':              d,
        'b':              b,
        'K_0':            K_0,
        'd_eff':          d_eff,
        'r_eff':          r_eff,
        'alpha':          alpha,
        'beta':           beta,
        'y_inf':          yi,
        'y0':             y_init,
        'y0_yinf_ratio':  y_init / max(yi, 1e-12),
    })

    if (i + 1) % 1000 == 0:
        print(f'  {i+1}/{N_SIMS} done')

df_out    = pd.concat(records, ignore_index=True)
df_params = pd.DataFrame(param_records)

# Column names match the original schema so metrics-processing.py runs unchanged
traj_path   = os.path.join(out_dir, 'simulated_scaled_trajectories.csv')
params_path = os.path.join(out_dir, 'simulated_parameters.csv')

df_out.to_csv(traj_path,   index=False)
df_params.to_csv(params_path, index=False)

n_failed  = int(df_out.groupby('Simulation_ID')['V_scaled'].apply(lambda v: v.isna().all()).sum())
n_clamped = int(df_params['y_frac_clamped'].sum())
total     = len(df_params)
print(f'\nN = {total} simulations   ({n_failed} ODE failures)')
print(f'y0 clamped to ≤2·y∞: {n_clamped} ({100*n_clamped/total:.1f}%)')
print(f'Classification → run metrics-processing.py for trajectory + analytical categories')
print()
print(f'Trajectories → {traj_path}')
print(f'Parameters   → {params_path}')

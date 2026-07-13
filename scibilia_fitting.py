import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import os
import time
import warnings

warnings.filterwarnings('ignore')

STUDY_NAME = sys.argv[1] if len(sys.argv) > 1 else 'Study3'

# Fixed parameters (removed from optimisation)
D2_FIXED     = 0.75   # D2, D3 — population median, low CV; D1 freed
D3_FIXED     = 0.75
SIGMA2_FIXED = 0.03   # sigma2   — weakly identified (z not observable)
MP_FIXED     = 1e-6   # mP_tilde — narrow log-IQR, both bounds hit

# y0 constraint: y0 <= Y0_CAP * y_inf
Y0_CAP = 2.0

# Blow-up guard: penalise trajectories that exceed this multiple of V(0)=1.
# Set high enough to allow progressive tumours but catch true divergences.
# v1 used 4× which was too tight; 50× only catches runaway solutions.
BLOWUP_CAP = 50.0

MIN_TROUGH_DAY = 7
REGROWTH_FRAC  = 0.20


def ode_system(t, state, params):
    x, y, z = max(0, state[0]), max(0, state[1]), max(0, state[2])

    r            = params['r']
    mu           = params['mu']
    mT_tilde     = params['mT_tilde']
    s_tilde      = params['s_tilde']
    p            = params['p']
    sigma1_tilde = params['sigma1_tilde']
    k            = params['k']
    D1           = params['D1']

    T_val      = max(1e-12, x)
    kill_denom = mT_tilde + T_val

    dxdt = (r * x) - (D1 * mu * y * x) / kill_denom

    dydt = (s_tilde
            + ((MP_FIXED + D2_FIXED * T_val) / (MP_FIXED + T_val))
            * (p * T_val * y) / kill_denom
            - sigma1_tilde * y**2
            - (1.0 - D3_FIXED) * (k * T_val * y) / (MP_FIXED + T_val))

    dzdt = ((1.0 - D3_FIXED) * (k * T_val * y) / (MP_FIXED + T_val)
            - SIGMA2_FIXED * z)

    return [dxdt, dydt, dzdt]


# Dropped:  phi (not in ODE)
# Fixed:    D2, D3, sigma2, mP_tilde   (D1 freed vs v2)
_ODE_PARAMS = ['r', 'mu', 'mT_tilde', 's_tilde', 'p', 'sigma1_tilde', 'k', 'D1']
_IC_PARAMS  = ['x0', 'y_frac']
param_names = _ODE_PARAMS + _IC_PARAMS
N_ODE       = len(_ODE_PARAMS)   # 8

is_log = [
    False, False, True,  True,  False, True,  False, False,  # r mu mT s p sig1 k D1
    False, False,                                              # x0 y_frac
]

bounds_lower = [
    0.0001, 0.01,  1e-6,  1e-8,  0.01, 0.001, 0.001, 0.5,
    0.50,   0.0,
]
bounds_upper = [
    0.1,    1.0,   1.0,   0.01,  1.0,  50.0,  1.0,   1.0,
    1.0,    1.0,
]

search_bounds_lower, search_bounds_upper = [], []
for lo, hi, log_flag in zip(bounds_lower, bounds_upper, is_log):
    if log_flag:
        search_bounds_lower.append(np.log10(lo))
        search_bounds_upper.append(np.log10(hi))
    else:
        search_bounds_lower.append(lo)
        search_bounds_upper.append(hi)


def _y_inf(p, k, sigma1_tilde, s_tilde):
    """Immune saturation limit as tumour burden x → ∞."""
    G_inf = D2_FIXED * p - (1.0 - D3_FIXED) * k
    disc  = G_inf**2 + 4.0 * sigma1_tilde * s_tilde
    return (G_inf + np.sqrt(max(0.0, disc))) / (2.0 * sigma1_tilde)


def _clamp_y_frac(y_frac, x0, p, k, sigma1_tilde, s_tilde):
    """Return y_frac clamped so that y0 = (1-x0)*y_frac <= Y0_CAP * y_inf."""
    yi       = _y_inf(p, k, sigma1_tilde, s_tilde)
    max_y0   = Y0_CAP * yi
    max_frac = min(1.0, max_y0 / max(1.0 - x0, 1e-10))
    return min(y_frac, max_frac), yi


def _first_phase_endpoint(t_orig, y_norm):
    t_grid = np.arange(float(t_orig.min()), float(t_orig.max()) + 1)
    y_grid = np.interp(t_grid, t_orig, y_norm)

    mask  = t_grid >= 0
    t_pos = t_grid[mask]
    y_pos = y_grid[mask]

    if len(y_pos) < 4:
        return float(t_orig.max())

    idx_trough = int(np.argmin(y_pos))
    t_trough   = float(t_pos[idx_trough])
    v_trough   = float(y_pos[idx_trough])

    if t_trough < MIN_TROUGH_DAY:
        return float(t_orig.max())

    y_after = y_pos[idx_trough:]
    if len(y_after) < 2:
        return float(t_orig.max())

    if float(np.max(y_after)) > v_trough * (1.0 + REGROWTH_FRAC):
        return t_trough

    return float(t_orig.max())


def objective_function(p_array, t_fit, y_fit):
    p_dict = {}
    for i in range(N_ODE):
        val = 10 ** p_array[i] if is_log[i] else p_array[i]
        p_dict[_ODE_PARAMS[i]] = val

    x0     = p_array[N_ODE]
    y_frac = p_array[N_ODE + 1]

    y_frac_eff, _ = _clamp_y_frac(
        y_frac, x0, p_dict['p'], p_dict['k'],
        p_dict['sigma1_tilde'], p_dict['s_tilde']
    )

    y0_init = [x0,
               (1.0 - x0) * y_frac_eff,
               (1.0 - x0) * (1.0 - y_frac_eff)]

    sol = solve_ivp(ode_system, [0, float(t_fit[-1]) + 1.0], y0_init,
                    args=(p_dict,), t_eval=t_fit, method='LSODA',
                    rtol=1e-4, atol=1e-7)

    if not sol.success or sol.y.shape[1] != len(t_fit):
        return np.full(len(t_fit), 1e6)

    v_pred = sol.y[0] + sol.y[1] + sol.y[2]

    if np.max(v_pred) > BLOWUP_CAP:
        return np.full(len(t_fit), 1e6)

    return v_pred - y_fit


def calc_metrics(y_true, y_pred, n_params):
    residuals = y_true - y_pred
    n      = len(y_true)
    rss    = float(np.sum(residuals ** 2))
    rmse   = np.sqrt(rss / n)
    mae    = float(np.mean(np.abs(residuals)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2     = 1.0 - rss / ss_tot if ss_tot > 1e-12 else np.nan
    aic    = n * np.log(rss / n) + 2.0 * n_params if rss > 0 and n > 0 else np.nan
    return rmse, r2, mae, aic


if __name__ == '__main__':
    NUM_STARTS        = 10     # restarts for every patient
    NUM_STARTS_RETRY  = 70     # additional restarts if R² < R2_RETRY_THRESHOLD
    R2_RETRY_THRESHOLD = 0.80  # patients below this get the extra restarts

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    data_path = os.path.join(script_dir, f'{STUDY_NAME}.csv')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f'\n[ERROR] Could not find {data_path}')
        exit()

    valid_ids = [pid for pid, grp in df.groupby('id')
                 if grp['normalized_volume'].notna().sum() >= 4]
    total_patients = len(valid_ids)
    print(f'Found {total_patients} patients in {STUDY_NAME} with ≥4 tumour scans.')

    # v2 param order: r mu mT_tilde s_tilde p sigma1_tilde k  x0 y_frac
    # v3 param order: r mu mT_tilde s_tilde p sigma1_tilde k D1 x0 y_frac
    # Warm start inserts D1=0.75 at position 7 and converts to search space.
    _v2_free = ['r', 'mu', 'mT_tilde', 's_tilde', 'p', 'sigma1_tilde', 'k', 'x0', 'y_frac']
    _v2_log  = [False, False, True, True, False, True, False, False, False]

    warm_starts: dict[int, list[float]] = {}
    v2_path = os.path.join(script_dir, 'fit_results_v2', f'{STUDY_NAME}_fit_results.csv')
    if os.path.exists(v2_path):
        v2_df = pd.read_csv(v2_path)
        for _, row in v2_df.iterrows():
            p0 = []
            for pname, log_flag in zip(_v2_free, _v2_log):
                val = row[pname]
                p0.append(np.log10(val) if log_flag else val)
            # insert D1 (was fixed at 0.75 in v2) at position 7 (after k)
            p0.insert(7, 0.75)
            # clamp to search bounds
            p0 = [np.clip(v, lo, hi) for v, lo, hi
                  in zip(p0, search_bounds_lower, search_bounds_upper)]
            warm_starts[int(row['id'])] = p0

    all_fit_results = []
    t_wall_start = time.time()

    for idx, pid in enumerate(valid_ids):
        grp      = df[df['id'] == pid].dropna(subset=['normalized_volume', 'Treatment_Day'])
        t_lesion = grp['Treatment_Day'].values
        y_lesion = grp['normalized_volume'].values
        d_lesion = grp['TargetLesionLongDiam_mm'].values

        if t_lesion.max() < 0:
            continue

        # V0 in cm³: diameter at t=0 (interpolated/extrapolated), sphere formula
        d_at_0 = np.interp(0, t_lesion, d_lesion)
        V0_cm3 = (np.pi / 6.0) * (d_at_0 / 10.0) ** 3

        vol_day0 = np.interp(0, t_lesion, y_lesion)
        y_lesion = y_lesion / vol_day0

        t_end     = _first_phase_endpoint(t_lesion, y_lesion)
        truncated = bool(t_end < t_lesion.max())

        t_all = np.arange(t_lesion.min(), t_lesion.max() + 1)
        y_all = np.interp(t_all, t_lesion, y_lesion)
        mask  = (t_all >= 0) & (t_all <= t_end)
        t_fit = t_all[mask].astype(float)
        y_fit = y_all[mask]

        trunc_tag = f' [clip t={t_end:.0f}]' if truncated else ''
        print(f'  [{idx+1:3d}/{total_patients}] id={pid}{trunc_tag} ...', end=' ', flush=True)
        t0 = time.time()

        best_cost     = float('inf')
        best_p_search = None

        starts = []
        if pid in warm_starts:
            starts.append(warm_starts[pid])
        starts += [[np.random.uniform(lo, hi)
                    for lo, hi in zip(search_bounds_lower, search_bounds_upper)]
                   for _ in range(NUM_STARTS - len(starts))]

        for p0 in starts:
            res = least_squares(
                objective_function,
                x0=p0,
                bounds=(search_bounds_lower, search_bounds_upper),
                args=(t_fit, y_fit),
                max_nfev=300,
                ftol=1e-5, gtol=1e-5, xtol=1e-5,
            )
            if res.cost < best_cost:
                best_cost     = res.cost
                best_p_search = res.x

        _bp_tmp = [10 ** v if is_log[i] else v for i, v in enumerate(best_p_search)]
        _pd_tmp = dict(zip(_ODE_PARAMS, _bp_tmp[:N_ODE]))
        _yfe, _ = _clamp_y_frac(_bp_tmp[N_ODE+1], _bp_tmp[N_ODE],
                                 _pd_tmp['p'], _pd_tmp['k'],
                                 _pd_tmp['sigma1_tilde'], _pd_tmp['s_tilde'])
        _ic_tmp = [_bp_tmp[N_ODE],
                   (1-_bp_tmp[N_ODE])*_yfe,
                   (1-_bp_tmp[N_ODE])*(1-_yfe)]
        _s_tmp  = solve_ivp(ode_system, [0, float(t_fit[-1])+1], _ic_tmp,
                            args=(_pd_tmp,), t_eval=t_fit, method='LSODA',
                            rtol=1e-4, atol=1e-7)
        if _s_tmp.success and _s_tmp.y.shape[1] == len(t_fit):
            _vp = _s_tmp.y[0] + _s_tmp.y[1] + _s_tmp.y[2]
            _ss = float(np.sum((y_fit - np.mean(y_fit))**2))
            _r2_tmp = 1 - float(np.sum((_vp - y_fit)**2)) / _ss if _ss > 1e-12 else -np.inf
        else:
            _r2_tmp = -np.inf

        if _r2_tmp < R2_RETRY_THRESHOLD:
            print(f'retry (R²={_r2_tmp:.3f}) ...', end=' ', flush=True)
            for _ in range(NUM_STARTS_RETRY):
                p0  = [np.random.uniform(lo, hi)
                       for lo, hi in zip(search_bounds_lower, search_bounds_upper)]
                res = least_squares(
                    objective_function,
                    x0=p0,
                    bounds=(search_bounds_lower, search_bounds_upper),
                    args=(t_fit, y_fit),
                    max_nfev=500,
                    ftol=1e-6, gtol=1e-6, xtol=1e-6,
                )
                if res.cost < best_cost:
                    best_cost     = res.cost
                    best_p_search = res.x

        elapsed = time.time() - t0
        print(f'done ({elapsed:.1f}s)')

        best_p = [10 ** v if is_log[i] else v for i, v in enumerate(best_p_search)]
        p_dict = dict(zip(_ODE_PARAMS, best_p[:N_ODE]))

        x0_best     = best_p[N_ODE]
        y_frac_best = best_p[N_ODE + 1]

        y_frac_eff, y_inf_val = _clamp_y_frac(
            y_frac_best, x0_best, p_dict['p'], p_dict['k'],
            p_dict['sigma1_tilde'], p_dict['s_tilde']
        )

        y0_init = [x0_best,
                   (1.0 - x0_best) * y_frac_eff,
                   (1.0 - x0_best) * (1.0 - y_frac_eff)]

        sol = solve_ivp(ode_system, [0, float(t_fit[-1]) + 10],
                        y0_init, args=(p_dict,), t_eval=t_fit,
                        method='LSODA', rtol=1e-6, atol=1e-9)

        if sol.success and sol.y.shape[1] == len(t_fit):
            v_pred = sol.y[0] + sol.y[1] + sol.y[2]
        else:
            v_pred = np.full(len(t_fit), np.nan)

        rmse, r2, mae, aic = calc_metrics(y_fit, v_pred, n_params=len(param_names))

        # cm³ metrics — scale residuals by V0 (R² is identical, AIC shifts with V0²)
        rmse_cm3, _, mae_cm3, aic_cm3 = calc_metrics(
            y_fit * V0_cm3, v_pred * V0_cm3, n_params=len(param_names)
        )

        record = {
            'id':              pid,
            'V0_cm3':          V0_cm3,
            # normalized-scale metrics
            'rmse_volume':     rmse,
            'r2_volume':       r2,
            'mae_volume':      mae,
            'aic':             aic,
            # cm³-scale metrics
            'rmse_cm3':        rmse_cm3,
            'mae_cm3':         mae_cm3,
            'aic_cm3':         aic_cm3,
            'truncated':       truncated,
            't_end':           t_end,
            'y_inf':           y_inf_val,
            'y0':              (1.0 - x0_best) * y_frac_eff,
            'y0_yinf_ratio':   (1.0 - x0_best) * y_frac_eff / max(y_inf_val, 1e-12),
            'y_frac_clamped':  y_frac_eff < y_frac_best,
        }
        for i, pname in enumerate(param_names):
            record[pname] = best_p[i]
        record['y_frac_eff'] = y_frac_eff
        record['D2']       = D2_FIXED
        record['D3']       = D3_FIXED
        record['sigma2']   = SIGMA2_FIXED
        record['mP_tilde'] = MP_FIXED

        all_fit_results.append(record)

    df_results = pd.DataFrame(all_fit_results)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else script_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{STUDY_NAME}_fit_results.csv')
    df_results.to_csv(out_path, index=False)
    print(f'  Results saved to: {out_path}')

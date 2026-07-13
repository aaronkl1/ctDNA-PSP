import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import os
import warnings

warnings.filterwarnings('ignore')

D2_FIX   = 0.75
D3_FIX   = 0.75
SIG2_FIX = 0.03
MP_FIX   = 1e-6
EPSILON  = 25.0                # ctDNA clearance, ~40 min half-life

Y0_CAP       = 2.0    # y0 ≤ Y0_CAP * y_inf
CTDNA_WEIGHT = 0.3
NUM_STARTS   = 25

# Minimum data requirements (excluding the t=0 baseline we add)
MIN_TUMOR_FU = 3   # ≥ 3 tumour follow-up scans  → ≥ 4 total
MIN_CTDNA_FU = 1   # ≥ 1 post-baseline ctDNA measurement

# (name, lo, hi, log_search)
PDEFS = [
    ('r',            0.001, 0.10,  False),
    ('mu',           0.01,  1.00,  False),
    ('mT_tilde',     1e-6,  1.00,  True ),
    ('s_tilde',      1e-8,  1e-2,  True ),
    ('p',            0.01,  1.00,  False),
    ('sigma1_tilde', 1e-3,  50.0,  True ),
    ('k',            0.001, 1.00,  False),
    ('D1',           0.50,  1.00,  False),
    ('x0',           0.50,  0.995, False),
    ('y_frac',       0.001, 0.999, False),
    ('phi',          0.05,  0.99,  False),
]

SLO = [np.log10(d[1]) if d[3] else d[1] for d in PDEFS]
SHI = [np.log10(d[2]) if d[3] else d[2] for d in PDEFS]


def decode(v):
    return {d[0]: (10.0 ** v[i] if d[3] else v[i])
            for i, d in enumerate(PDEFS)}


def y_inf_val(p):
    """Immune saturation limit as x → ∞ (fixed D2, D3, mP→0)."""
    G    = D2_FIX * p['p'] - (1.0 - D3_FIX) * p['k']
    disc = G**2 + 4.0 * p['sigma1_tilde'] * p['s_tilde']
    return max(0.0, (G + np.sqrt(max(0.0, disc))) / (2.0 * p['sigma1_tilde']))


def make_ics(p):
    """
    Compute initial conditions and ctDNA coefficients.

    y0 is clamped to Y0_CAP * y_inf.
    alpha, beta are set so that C(0) = 1 satisfies the v3 quasi-steady-state:
        dC/dt|_{t=0} = r_eff · C(0)      (ctDNA grows at the net tumour rate)
    """
    x0 = p['x0']
    yi = y_inf_val(p)
    max_yf = min(0.9999, Y0_CAP * yi / max(1.0 - x0, 1e-10))
    yf = min(p['y_frac'], max_yf)
    y0 = (1.0 - x0) * yf
    z0 = (1.0 - x0) - y0

    r   = p['r'];  phi = p['phi']
    d   = r * phi / (1.0 - phi)
    b   = r + d
    K0  = p['D1'] * p['mu'] * y0 / (p['mT_tilde'] + x0)
    de  = d + K0                       # d_eff (recorded only)
    re  = r - K0                       # r_eff (recorded only)
    alpha = (EPSILON + r) / x0
    beta  = (p['mu'] / max(d, 1e-12)) * (EPSILON + r) / x0

    return x0, y0, z0, 1.0, alpha, beta, {'d': d, 'b': b, 'K0': K0,
                                           'd_eff': de, 'r_eff': re,
                                           'y_frac_used': yf, 'y_inf': yi}


def ode_rhs(t, state, p, alpha, beta):
    x, y, z, c = [max(0.0, s) for s in state]
    T  = max(1e-12, x)
    kd = p['mT_tilde'] + T
    mp = MP_FIX + T
    kill   = p['D1'] * p['mu'] * y * T / kd
    inhib  = (MP_FIX + D2_FIX * T) / mp
    prolif = p['p'] * T * y / kd
    exh    = (1.0 - D3_FIX) * p['k'] * T * y / mp
    return [
        p['r'] * x - kill,
        p['s_tilde'] + inhib * prolif - p['sigma1_tilde'] * y**2 - exh,
        exh - SIG2_FIX * z,
        alpha * x + beta * (kill / p['mu']) - EPSILON * c,
    ]


def run_ode(p, t_span, *, t_eval=None, dense=False):
    x0, y0, z0, c0, al, be, _ = make_ics(p)
    return solve_ivp(ode_rhs, t_span, [x0, y0, z0, c0],
                     args=(p, al, be), t_eval=t_eval, dense_output=dense,
                     method='Radau', rtol=1e-5, atol=1e-8)


def objective(v, t_V, V_obs, t_C, C_obs):
    p = decode(v)
    x0, y0, z0, c0, al, be, _ = make_ics(p)
    t_max = max(t_V[-1] if len(t_V) else 0.0,
                t_C[-1] if len(t_C) else 0.0) + 1.0
    try:
        sol = solve_ivp(ode_rhs, [0.0, t_max], [x0, y0, z0, c0],
                        args=(p, al, be), dense_output=True,
                        method='Radau', rtol=1e-4, atol=1e-7)
    except Exception:
        return np.ones(300) * 10.0

    res = []
    if len(t_V) >= 2:
        td  = np.arange(int(t_V[0]), int(t_V[-1]) + 1, dtype=float)
        Vt  = np.interp(td, t_V, V_obs)
        st  = sol.sol(td)
        res.extend(np.clip(st[0] + st[1] + st[2], 0.0, None) - Vt)
    if len(t_C) >= 2:
        td  = np.arange(int(t_C[0]), int(t_C[-1]) + 1, dtype=float)
        Ct  = np.interp(td, t_C, C_obs)
        st  = sol.sol(td)
        res.extend((np.clip(st[3], 0.0, None) - Ct) * CTDNA_WEIGHT)
    return np.array(res) if res else np.array([1e-6])


def _G2d(x, p):
    return ((p['mP_tilde'] + D2_FIX * x) / (p['mP_tilde'] + x)
            * p['p'] * x / (p['mT_tilde'] + x)
            - (1.0 - D3_FIX) * p['k'] * x / (p['mP_tilde'] + x))


def _ode_2d(t, s, p):
    x, y = max(0.0, s[0]), max(0.0, s[1])
    return [p['r'] * x - p['D1'] * p['mu'] * y * x / (p['mT_tilde'] + x),
            p['s_tilde'] + _G2d(x, p) * y - p['sigma1_tilde'] * y**2]


def _run_2d_forward(x0, y0, p, T=600.0, esc=5.0, ctrl=0.05):
    def ev_e(t, s, *_): return s[0] - esc
    def ev_c(t, s, *_): return s[0] - ctrl
    ev_e.terminal = ev_c.terminal = True
    ev_e.direction = 1; ev_c.direction = -1
    try:
        sol = solve_ivp(_ode_2d, [0.0, T], [x0, y0], args=(p,),
                        events=[ev_e, ev_c], method='RK45',
                        rtol=1e-4, atol=1e-6, max_step=2.0)
        xe = sol.y[0, -1]
        if xe >= esc:  return 'escape'
        if xe <= ctrl: return 'control'
        w = max(1, sol.y.shape[1] // 10)
        return 'control' if sol.y[0, -1] < sol.y[0, -w] else 'escape'
    except Exception:
        return None


def analytical_cat(x0, y0, p):
    pp = dict(p); pp['mP_tilde'] = MP_FIX
    outcome = _run_2d_forward(x0, y0, pp)
    if outcome == 'escape':
        return 'Progressor'
    dxdt = pp['r'] * x0 - pp['D1'] * pp['mu'] * y0 * x0 / (pp['mT_tilde'] + x0)
    dydt = pp['s_tilde'] + _G2d(x0, pp) * y0 - pp['sigma1_tilde'] * y0**2
    if dxdt + dydt <= 0: return 'Responder'
    if dxdt <= 0:        return 'Immune_Pseudoprogressor'
    return 'Delayed_Pseudoprogressor'


def model_trajectory_cat(p, t_max=600.0, thresh=1.7):
    """Trajectory-based category from long-run model integration."""
    try:
        t_eval = np.arange(0.0, t_max + 1.0)
        sol = run_ode(p, [0.0, t_max], t_eval=t_eval)
        if not sol.success:
            return 'Unknown', np.nan, sol.y[0] + sol.y[1] + sol.y[2]
        V = np.clip(sol.y[0] + sol.y[1] + sol.y[2], 0.0, None)
        cross = np.where(V >= thresh)[0]
        if len(cross) == 0:
            return 'Responder', np.nan, V
        i_cross = cross[0]
        if i_cross + 1 >= len(V):
            return 'Progressor', t_eval[i_cross], V
        # Check if it comes back down after crossing
        V_after = V[i_cross:]
        if V_after.min() < V[i_cross]:
            return 'Pseudoprogressor', t_eval[i_cross], V
        return 'Progressor', t_eval[i_cross], V
    except Exception:
        return 'Unknown', np.nan, np.array([])


def calc_metrics(y_true, y_pred):
    res  = y_true - y_pred
    rmse = np.sqrt(np.mean(res**2))
    ss_t = np.sum((y_true - y_true.mean())**2)
    r2   = 1.0 - np.sum(res**2) / ss_t if ss_t > 1e-12 else np.nan
    return float(rmse), float(r2)


def load_patient_data(base):
    """
    Load xlsx, apply diameter→volume conversion, return per-patient data dict.
    Baseline (t=0, V=1, C=1) is prepended to each patient's series.
    """
    df_t = pd.read_excel(os.path.join(base, 'tumor-changes.xlsx'))
    df_c = pd.read_excel(os.path.join(base, 'ctDNA-changes.xlsx'))

    t_raw, c_raw = {}, {}

    for pid, g in df_t.groupby('ID'):
        g = g.sort_values('weeks_on_trial')
        t = g['weeks_on_trial'].values * 7.0
        V = ((100.0 + g['change_target_lesion'].values) / 100.0) ** 3
        t_raw[pid] = (np.concatenate([[0.0], t]),
                      np.concatenate([[1.0], np.clip(V, 0.0, None)]))

    for pid, g in df_c.groupby('ID'):
        g = g.sort_values('weeks_on_trial')
        t = g['weeks_on_trial'].values * 7.0
        C = (100.0 + g['change_ctDNA'].values) / 100.0
        if t[0] > 0.5:          # no week-0 row in file → prepend baseline
            t = np.concatenate([[0.0], t])
            C = np.concatenate([[1.0], C])
        c_raw[pid] = (t, np.clip(C, 0.0, None))

    valid = {}
    for pid in sorted(set(t_raw) & set(c_raw)):
        t_V, V = t_raw[pid]; t_C, C = c_raw[pid]
        n_tu_fu = len(t_V) - 1                    # followup scans only
        n_ct_fu = int((t_C > 0.5).sum())          # post-baseline ctDNA
        if n_tu_fu >= MIN_TUMOR_FU and n_ct_fu >= MIN_CTDNA_FU:
            valid[pid] = {'t_V': t_V, 'V': V, 't_C': t_C, 'C': C}

    return valid


def make_plot(pid, p, t_plot, states_plot, t_V, V, t_C, C,
              rmse_V, r2_V, rmse_C, r2_C, acat, mcat, plot_dir):
    x_p, y_p, z_p, c_p = np.clip(states_plot, 0.0, None)
    V_p = x_p + y_p + z_p

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(t_plot, V_p,  'k-',  lw=2.0,  label='V = x+y+z')
    ax.plot(t_plot, x_p,  'r--', lw=1.2,  label='x (tumour)')
    ax.plot(t_plot, y_p,  'g:',  lw=1.2,  label='y (active immune)')
    ax.plot(t_plot, z_p,  'b-.', lw=1.2,  label='z (exhausted)')
    ax.scatter(t_V, V,    c='k',  s=55, zorder=6, label='Data (volume)')
    ax.axhline(1.7, color='gray', lw=0.8, ls=':', alpha=0.7, label='PSP threshold')
    ax.set_xlabel('Days'); ax.set_ylabel('Normalised V')
    ax.set_title(f'Tumour volume\nRMSE={rmse_V:.3f}  R²={r2_V:.3f}')
    ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t_plot, c_p,  'm-',  lw=2.0, label='C (model)')
    ax2.scatter(t_C, C,    c='m', edgecolors='k', s=55, zorder=6, label='Data (ctDNA)')
    ax2.set_xlabel('Days'); ax2.set_ylabel('Normalised C')
    ax2.set_title(f'ctDNA\nRMSE={rmse_C:.3f}  R²={r2_C:.3f}')
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{pid}  |  Analytical: {acat}  |  Model trajectory: {mcat}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{pid}_fit_v3.png'), dpi=200,
                bbox_inches='tight')
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir    = os.path.join(script_dir, 'output_v3')
    plot_dir   = os.path.join(out_dir, 'fit_plots_v3')
    os.makedirs(plot_dir, exist_ok=True)

    np.random.seed(42)

    print('Loading patient data from xlsx files...')
    patients = load_patient_data(script_dir)
    print(f'  {len(patients)} patients meet data criteria '
          f'(≥{MIN_TUMOR_FU} tumour FU, ≥{MIN_CTDNA_FU} ctDNA FU)')

    rows = []
    for pid, d in sorted(patients.items()):
        for t, v in zip(d['t_V'], d['V']):
            rows.append({'ID': pid, 'time_days': t, 'V_normalized': v,
                         'modality': 'tumor'})
        for t, c in zip(d['t_C'], d['C']):
            rows.append({'ID': pid, 'time_days': t, 'C_normalized': c,
                         'modality': 'ctDNA'})
    data_csv = os.path.join(script_dir, 'bratman-data-v3.csv')
    pd.DataFrame(rows).to_csv(data_csv, index=False)
    print(f'  Saved {data_csv}')

    results = []
    for pid in sorted(patients):
        d   = patients[pid]
        t_V, V = d['t_V'], d['V']
        t_C, C = d['t_C'], d['C']
        print(f'\nFitting {pid}  '
              f'(tumour={len(t_V)} pts, ctDNA={len(t_C)} pts)...')

        best_cost, best_v = np.inf, None
        for i in range(NUM_STARTS):
            v0 = [np.random.uniform(lo, hi) for lo, hi in zip(SLO, SHI)]
            try:
                res = least_squares(objective, v0,
                                    bounds=(SLO, SHI),
                                    args=(t_V, V, t_C, C),
                                    method='trf', max_nfev=3000,
                                    ftol=1e-7, xtol=1e-7)
                if res.cost < best_cost:
                    best_cost, best_v = res.cost, res.x
            except Exception:
                pass
            if (i + 1) % 5 == 0:
                print(f'  start {i+1}/{NUM_STARTS} → best cost {best_cost:.4f}')

        if best_v is None:
            print(f'  [SKIP] optimisation failed for {pid}')
            continue

        p = decode(best_v)
        x0, y0, z0, c0, al, be, extra = make_ics(p)

        t_max_obs = max(t_V[-1], t_C[-1])
        t_max_ext = t_max_obs + 200.0
        t_eval_p  = np.linspace(0.0, t_max_ext, 600)
        sol = run_ode(p, [0.0, t_max_ext], t_eval=t_eval_p, dense=True)

        if not sol.success:
            print(f'  [WARN] dense ODE failed for {pid}')
            continue

        st_V = sol.sol(t_V); V_pred = np.clip(st_V[0]+st_V[1]+st_V[2], 0, None)
        st_C = sol.sol(t_C); C_pred = np.clip(st_C[3], 0, None)
        rmse_V, r2_V = calc_metrics(V, V_pred)
        rmse_C, r2_C = calc_metrics(C, C_pred)

        acat = analytical_cat(x0, y0, p)
        mcat, t_prog_model, _ = model_trajectory_cat(p, t_max=600.0)

        yi       = extra['y_inf']
        yf_used  = extra['y_frac_used']
        y0_yinf  = y0 / max(yi, 1e-12)

        print(f'  cost={best_cost:.4f}  '
              f'RMSE_V={rmse_V:.3f}  R²_V={r2_V:.3f}  '
              f'RMSE_C={rmse_C:.3f}  R²_C={r2_C:.3f}  '
              f'Anal={acat}  Model={mcat}')

        make_plot(pid, p, t_eval_p, sol.sol(t_eval_p),
                  t_V, V, t_C, C,
                  rmse_V, r2_V, rmse_C, r2_C,
                  acat, mcat, plot_dir)

        rec = {
            'ID':                   pid,
            'rmse_V':               rmse_V,   'r2_V': r2_V,
            'rmse_C':               rmse_C,   'r2_C': r2_C,
            'Analytical_Category':  acat,
            'Model_Category':       mcat,
            'T_prog_model_days':    t_prog_model,
            # initial conditions
            'x0': x0, 'y0': y0, 'z0': z0,
            'y_frac': yf_used, 'y_inf': yi, 'y0_over_y_inf': y0_yinf,
            # ctDNA derived
            'd_eff': extra['d_eff'], 'r_eff': extra['r_eff'],
            'K0':    extra['K0'],
        }
        # Free fitted parameters
        for name, *_ in PDEFS:
            rec[name] = p[name]
        # Fixed parameters (for completeness)
        rec.update({'D2': D2_FIX, 'D3': D3_FIX, 'sigma2': SIG2_FIX,
                    'mP_tilde': MP_FIX, 'epsilon': EPSILON})
        results.append(rec)

    df_res = pd.DataFrame(results)
    res_path = os.path.join(out_dir, 'fit_results_v3.csv')
    df_res.to_csv(res_path, index=False)
    print(f'\n{"=" * 60}')
    print(f'Saved: {res_path}')
    print(f'Plots: {plot_dir}')
    print(f'{"=" * 60}')

    # Quick summary
    print('\nAnalytical category summary:')
    print(df_res['Analytical_Category'].value_counts().to_string())
    print('\nModel trajectory category summary:')
    print(df_res['Model_Category'].value_counts().to_string())
    print(f'\nMedian RMSE_V : {df_res["rmse_V"].median():.3f}')
    print(f'Median RMSE_C : {df_res["rmse_C"].median():.3f}')
    print(f'Median R²_V   : {df_res["r2_V"].median():.3f}')
    print(f'Median R²_C   : {df_res["r2_C"].median():.3f}')


if __name__ == '__main__':
    main()

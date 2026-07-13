import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

date_str = datetime.today().strftime('%m-%d')
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

base_dir = os.path.join(script_dir, f"{date_str}_phase")
os.makedirs(base_dir, exist_ok=True)

params = {
    'r': 0.015, 'mu': 0.8, 'mT': 1e6, 'mP': 1e6,
    'p': 1.0, 'k3': 0.1, 's': 1e4, 'sigma1': 1e-8,
    'D1': 0.8, 'D2': 0.8, 'D3': 0.8
}

def ode(t, state):
    x, y = max(0, state[0]), max(0, state[1])
    dxdt = params['r']*x - params['D1']*params['mu']*y*x / (params['mT'] + x)
    G = ((params['mP'] + params['D2']*x) / (params['mP'] + x)) * (params['p']*x)/(params['mT'] + x) \
        - (1.0 - params['D3']) * params['k3']*x/(params['mP'] + x)
    dydt = params['s'] + G*y - params['sigma1']*(y**2)
    return [dxdt, dydt]

def ode_back(t, state):
    res = ode(t, state)
    return [-res[0], -res[1]]

x_max = 5e7
y_max = 2e8
x_vals = np.linspace(0, x_max, 1000)

y_tumor_null = (params['r'] / (params['D1'] * params['mu'])) * x_vals \
             + (params['r'] * params['mT'] / (params['D1'] * params['mu']))

G_vals = ((params['mP'] + params['D2']*x_vals) / (params['mP'] + x_vals)) \
         * (params['p']*x_vals)/(params['mT'] + x_vals) \
         - (1.0 - params['D3']) * params['k3']*x_vals/(params['mP'] + x_vals)
y_immune_null = (G_vals + np.sqrt(G_vals**2 + 4*params['sigma1']*params['s'])) \
              / (2*params['sigma1'])

kill_term = params['D1'] * params['mu'] * x_vals / (params['mT'] + x_vals)
L_term_a  = G_vals - kill_term
y_L_null  = (L_term_a + np.sqrt(L_term_a**2 + 4*params['sigma1']*(params['r']*x_vals + params['s']))) \
           / (2*params['sigma1'])

y_tumor_null_plot = np.where(y_tumor_null > y_max*1.5, np.nan, y_tumor_null)
y_L_null_plot     = np.where(y_L_null     > y_max*1.5, np.nan, y_L_null)
y_L_null_safe     = np.where(np.isnan(y_L_null), y_max*2, y_L_null)

diff = y_immune_null - y_tumor_null
idx  = np.where(np.diff(np.sign(diff)))[0]

sep_pts_x, sep_pts_y = [], []
if len(idx) > 0:
    x_eq = x_vals[idx[0]]
    y_eq = y_immune_null[idx[0]]
    epsilon   = x_max * 1e-4
    sol_back  = solve_ivp(ode_back, [0, 5000], [x_eq - epsilon, y_eq + epsilon], max_step=1.0)
    sep_pts_x = sol_back.y[0]
    sep_pts_y = sol_back.y[1]
    valid_idx = (sep_pts_x >= 0) & (sep_pts_y >= 0)
    sep_pts_x = sep_pts_x[valid_idx]
    sep_pts_y = sep_pts_y[valid_idx]
    sort_idx  = np.argsort(sep_pts_x)
    sep_pts_x = sep_pts_x[sort_idx]
    sep_pts_y = sep_pts_y[sort_idx]

y_sep_vals = np.zeros_like(x_vals)
if len(sep_pts_x) > 1:
    sep_interp = interp1d(sep_pts_x, sep_pts_y, bounds_error=False,
                          fill_value=(sep_pts_y[0], 0))
    y_sep_vals = sep_interp(x_vals)
    y_sep_vals[x_vals > sep_pts_x[-1]] = 0

plt.figure(figsize=(10, 8))

plt.fill_between(x_vals, np.maximum(y_L_null_safe, y_sep_vals), y_max,
                 where=(y_max > np.maximum(y_L_null_safe, y_sep_vals)),
                 color='limegreen', alpha=0.25, label='Responder Zone (-1)', zorder=1)
plt.fill_between(x_vals, np.maximum(y_tumor_null, y_sep_vals), y_L_null_safe,
                 where=(y_L_null_safe > np.maximum(y_tumor_null, y_sep_vals)),
                 color='dodgerblue', alpha=0.35, label='Inflammation Pseudo (0I)', zorder=1)
plt.fill_between(x_vals, y_sep_vals, y_tumor_null,
                 where=(y_tumor_null > y_sep_vals),
                 color='mediumpurple', alpha=0.35, label='Delayed Pseudo (0D)', zorder=1)
plt.fill_between(x_vals, 0, y_sep_vals,
                 where=(y_sep_vals > 0),
                 color='salmon', alpha=0.25, label='Progressor Zone (1)', zorder=1)

plt.plot(x_vals, y_tumor_null_plot, 'r--', lw=2.5, label='Tumor Nullcline ($dx/dt=0$)', zorder=3)
plt.plot(x_vals, y_immune_null,     'b--', lw=2.5, label='Immune Nullcline ($dy/dt=0$)', zorder=3)
plt.plot(x_vals, y_L_null_plot,     'k:',  lw=2.5, label='Lesion Nullcline ($d(x+y)/dt=0$)', zorder=3)

if len(sep_pts_x) > 1:
    plt.plot(sep_pts_x, sep_pts_y, 'k-', lw=3, label='Separatrix', zorder=4)

plt.xlabel("Tumor Cells ($x$)", fontsize=13)
plt.ylabel("Active Immune Cells ($y$)", fontsize=13)
plt.xlim(0, x_max)
plt.ylim(0, y_max)
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(base_dir, 'phase_plane.png'), dpi=300)
plt.show()

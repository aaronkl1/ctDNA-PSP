import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    cohen_kappa_score, f1_score, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, 'output_v3')
FUZZ_DIR   = os.path.join(OUT_DIR, 'fuzzed_comparisons', 'y0_kappa')
os.makedirs(FUZZ_DIR, exist_ok=True)

PROG_THRESHOLD = 1.7
RESP_THRESHOLD = 0.343
T_MAX          = 300
INTERVAL_LO    = 25
INTERVAL_HI    = 50
MAX_OBS        = 14
RANDOM_STATE   = 42
TEST_SIZE      = 0.30

T_COLS    = [f't_{i}' for i in range(MAX_OBS)]
V_COLS    = [f'V_{i}' for i in range(MAX_OBS)]
C_COLS    = [f'C_{i}' for i in range(MAX_OBS)]
FEATS     = T_COLS + V_COLS + C_COLS
FEATS_Y0  = FEATS + ['y0']

print("Loading raw trajectories...")
raw = pd.read_csv(os.path.join(OUT_DIR, 'simulated_scaled_trajectories.csv'))

np.random.seed(RANDOM_STATE)   # must match fuzzed_recist_comparisons.py exactly

print(f"Building fuzzed schedules ({INTERVAL_LO}–{INTERVAL_HI} d intervals)...")
rows = []
for sid, grp in raw.groupby('Simulation_ID'):
    t_arr = grp['Time'].values
    V_arr = grp['V_scaled'].values
    C_arr = grp['C_scaled'].values
    Y_arr = grp['Y_scaled'].values

    y0 = float(np.interp(0.0, t_arr, Y_arr))

    schedule = [0.0]
    while schedule[-1] < T_MAX + INTERVAL_HI:
        schedule.append(schedule[-1] + np.random.uniform(INTERVAL_LO, INTERVAL_HI))
    schedule = np.array(schedule)

    sched_obs = schedule[schedule <= T_MAX]
    V_samp    = np.interp(sched_obs, t_arr, V_arr)
    C_samp    = np.interp(sched_obs, t_arr, C_arr)

    prog_idx = np.where(V_samp >= PROG_THRESHOLD)[0]
    if len(prog_idx) > 0:
        i_prog      = prog_idx[0]
        t_prog_samp = float(sched_obs[i_prog])
        V_prog      = float(V_samp[i_prog])
        candidates  = schedule[schedule > t_prog_samp]
        if len(candidates) > 0:
            t_next = float(candidates[0])
            if t_next <= T_MAX:
                V_next = float(np.interp(t_next, t_arr, V_arr))
            else:
                t_next = np.nan
                V_next = np.nan
        else:
            t_next = np.nan
            V_next = np.nan
        later_V = V_samp[i_prog + 1:]
        label   = 'PSP' if (len(later_V) > 0 and np.any(later_V < PROG_THRESHOLD)) else 'Prog'
    else:
        t_prog_samp = np.nan
        V_prog      = np.nan
        t_next      = np.nan
        V_next      = np.nan
        label = 'Resp' if np.any(V_samp <= RESP_THRESHOLD) else 'SD'

    keep  = sched_obs <= t_prog_samp + 1e-6 if not np.isnan(t_prog_samp) \
            else np.ones(len(sched_obs), dtype=bool)
    t_obs = sched_obs[keep];  V_obs = V_samp[keep];  C_obs = C_samp[keep]
    n_obs = min(len(t_obs), MAX_OBS)

    t_feat = np.full(MAX_OBS, np.nan)
    V_feat = np.full(MAX_OBS, np.nan)
    C_feat = np.full(MAX_OBS, np.nan)
    t_feat[:n_obs] = t_obs[:n_obs]
    V_feat[:n_obs] = V_obs[:n_obs]
    C_feat[:n_obs] = C_obs[:n_obs]

    row = {'Simulation_ID': sid, 'label': label,
           't_prog_samp': t_prog_samp, 'V_prog': V_prog,
           't_next': t_next, 'V_next': V_next, 'y0': y0}
    for i in range(MAX_OBS):
        row[T_COLS[i]] = t_feat[i]
        row[V_COLS[i]] = V_feat[i]
        row[C_COLS[i]] = C_feat[i]
    rows.append(row)

df = pd.DataFrame(rows)
df['label3']   = df['label'].map({'Prog': 'Prog', 'PSP': 'PSP',
                                   'Resp': 'RSD',  'SD':  'RSD'})
df['label2pr'] = df['label'].map({'Prog': 'Prog', 'PSP': 'RSD',
                                   'Resp': 'RSD',  'SD':  'RSD'})
print(f"  {len(df)} patients | {df['label'].value_counts().to_dict()}")


def predict_recist(df_sub):
    return np.where(df_sub['t_prog_samp'].notna(), 'Prog', 'RSD')


def predict_irecist(df_sub, remap_psp=False):
    preds = []
    for _, row in df_sub.iterrows():
        if pd.isna(row['t_prog_samp']):
            preds.append('RSD')
        elif pd.isna(row['V_next']) or row['V_next'] >= PROG_THRESHOLD:
            preds.append('Prog')
        else:
            preds.append('RSD' if remap_psp else 'PSP')
    return np.array(preds)


def make_rf():
    return Pipeline([
        ('imp', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('clf', RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                        class_weight='balanced', n_jobs=-1)),
    ])


def safe_kappa(y_true, y_pred):
    try:
        return cohen_kappa_score(y_true, y_pred)
    except Exception:
        return 0.0


TASKS = {
    '3class': {
        'title':  '3-class (Prog / PSP / RSD)',
        'classes': ['Prog', 'PSP', 'RSD'],
        'y_col':  'label3',
        'filter': None,
        'psp_guard':     False,
        'irecist_remap': False,
    },
    '2class_prog_psp': {
        'title':  '2-class (Prog vs PSP)',
        'classes': ['Prog', 'PSP'],
        'y_col':  'label',
        'filter': ['Prog', 'PSP'],
        'psp_guard':     True,
        'irecist_remap': False,
    },
    '2class_prog_rsd': {
        'title':  '2-class (Prog vs RSD)',
        'classes': ['Prog', 'RSD'],
        'y_col':  'label2pr',
        'filter': None,
        'psp_guard':     False,
        'irecist_remap': True,
    },
}

METHOD_ORDER  = ['RECIST 1.1', 'iRECIST', 'RF', r'RF + $y_0$']
METHOD_COLORS = {
    'RECIST 1.1':  '#7f8c8d',
    'iRECIST':     '#2980b9',
    'RF':          '#e67e22',
    r'RF + $y_0$': '#27ae60',
}


def _savefig(fname_base):
    for ext in ('png', 'pdf'):
        plt.savefig(os.path.join(FUZZ_DIR, fname_base + '.' + ext),
                    dpi=300, bbox_inches='tight')


print("\n" + "=" * 60)
results_all = {}

for stem, cfg in TASKS.items():
    print(f"\n>>> {cfg['title']}")
    df_task = df[df['label'].isin(cfg['filter'])].copy() if cfg['filter'] else df.copy()
    y = df_task[cfg['y_col']]

    X_base = df_task[FEATS]
    X_tr_b, X_te_b, y_tr, y_te = train_test_split(
        X_base, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    X_tr_y0 = df_task.loc[y_tr.index, FEATS_Y0]
    X_te_y0 = df_task.loc[y_te.index, FEATS_Y0]
    df_te   = df_task.loc[y_te.index]

    recist_pred  = predict_recist(df_te)
    irecist_pred = predict_irecist(df_te, remap_psp=cfg['irecist_remap'])
    if cfg['psp_guard']:
        irecist_pred = np.where(irecist_pred == 'RSD', 'Prog', irecist_pred)

    rf_pipe = make_rf();  rf_pipe.fit(X_tr_b, y_tr)
    rf_pred = rf_pipe.predict(X_te_b)

    rf_y0_pipe = make_rf();  rf_y0_pipe.fit(X_tr_y0, y_tr)
    rf_y0_pred = rf_y0_pipe.predict(X_te_y0)

    preds = {
        'RECIST 1.1':   recist_pred,
        'iRECIST':      irecist_pred,
        'RF':           rf_pred,
        r'RF + $y_0$':  rf_y0_pred,
    }

    task_metrics = {}
    for mname, pred in preds.items():
        k    = safe_kappa(y_te, pred)
        f1m  = f1_score(y_te, pred, labels=cfg['classes'], average='macro', zero_division=0)
        bacc = balanced_accuracy_score(y_te, pred)
        mcc  = matthews_corrcoef(y_te, pred)
        task_metrics[mname] = {'kappa': k, 'f1_mac': f1m, 'bacc': bacc, 'mcc': mcc}
        print(f"  {mname:<16}  κ={k:.3f}  macro-F1={f1m:.3f}  BA={bacc:.3f}  MCC={mcc:.3f}")

    results_all[stem] = (cfg['title'], task_metrics)


for stem, (title, task_metrics) in results_all.items():
    vals   = [task_metrics[m]['kappa'] for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    lo_lim = min(-0.05, min(vals) - 0.05)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(METHOD_ORDER, vals, color=colors, alpha=0.88, width=0.5,
                  edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.018 if val >= 0 else bar.get_height() - 0.07
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(lo_lim, 1.13)
    ax.set_ylabel("Cohen's $\\kappa$", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.35)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _savefig(f'kappa_{stem}')
    plt.close()
    print(f"  Saved: kappa_{stem}.*")


fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

for ax, (stem, (title, task_metrics)) in zip(axes, results_all.items()):
    vals   = [task_metrics[m]['kappa'] for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    lo_lim = min(-0.05, min(vals) - 0.05)
    bars = ax.bar(METHOD_ORDER, vals, color=colors, alpha=0.88, width=0.5,
                  edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.018 if val >= 0 else bar.get_height() - 0.07
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_ylim(lo_lim, 1.13)
    ax.set_ylabel("Cohen's $\\kappa$", fontsize=11)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
    ax.tick_params(axis='x', labelsize=9, rotation=15)
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.35)
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
_savefig('kappa_combined_panel')
plt.close()
print("\nSaved: kappa_combined_panel.*")
print(f"\nAll figures → {FUZZ_DIR}/")

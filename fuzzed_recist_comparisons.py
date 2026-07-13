import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(SCRIPT_DIR, 'output_v3')
FUZZ_DIR     = os.path.join(OUT_DIR, 'fuzzed_comparisons')
os.makedirs(FUZZ_DIR, exist_ok=True)

PROG_THRESHOLD = 1.7
RESP_THRESHOLD = 0.343   # 0.7³ (30% diameter decrease)
T_MAX          = 300
INTERVAL_LO    = 25
INTERVAL_HI    = 50
MAX_OBS        = 14      # 300/25 = 12 steps + t=0 + margin
RANDOM_STATE   = 42
TEST_SIZE      = 0.30

T_COLS = [f't_{i}' for i in range(MAX_OBS)]
V_COLS = [f'V_{i}' for i in range(MAX_OBS)]
C_COLS = [f'C_{i}' for i in range(MAX_OBS)]
FEATS  = T_COLS + V_COLS + C_COLS   # 42 features

print("Loading raw trajectories...")
raw = pd.read_csv(os.path.join(OUT_DIR, 'simulated_scaled_trajectories.csv'))

np.random.seed(RANDOM_STATE)

print(f"Building fuzzed schedules ({INTERVAL_LO}–{INTERVAL_HI} d intervals)...")
rows = []
for sid, grp in raw.groupby('Simulation_ID'):
    t_arr = grp['Time'].values
    V_arr = grp['V_scaled'].values
    C_arr = grp['C_scaled'].values

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
        t_prog_samp = np.nan; V_prog = np.nan; t_next = np.nan; V_next = np.nan
        label = 'Resp' if np.any(V_samp <= RESP_THRESHOLD) else 'SD'

    keep  = sched_obs <= t_prog_samp + 1e-6 if not np.isnan(t_prog_samp) \
            else np.ones(len(sched_obs), dtype=bool)
    t_obs = sched_obs[keep]; V_obs = V_samp[keep]; C_obs = C_samp[keep]
    n_obs = min(len(t_obs), MAX_OBS)

    t_feat = np.full(MAX_OBS, np.nan)
    V_feat = np.full(MAX_OBS, np.nan)
    C_feat = np.full(MAX_OBS, np.nan)
    t_feat[:n_obs] = t_obs[:n_obs]
    V_feat[:n_obs] = V_obs[:n_obs]
    C_feat[:n_obs] = C_obs[:n_obs]

    row = {'Simulation_ID': sid, 'label': label,
           't_prog_samp': t_prog_samp, 'V_prog': V_prog,
           't_next': t_next, 'V_next': V_next}
    for i in range(MAX_OBS):
        row[T_COLS[i]] = t_feat[i]
        row[V_COLS[i]] = V_feat[i]
        row[C_COLS[i]] = C_feat[i]
    rows.append(row)

df = pd.DataFrame(rows)
n_prog = (df['label'] == 'Prog').sum()
n_psp  = (df['label'] == 'PSP').sum()
n_resp = (df['label'].isin(['Resp', 'SD'])).sum()
print(f"  {len(df)} patients  |  Prog={n_prog}  PSP={n_psp}  Resp/SD={n_resp}")

df['label3']   = df['label'].map({'Prog': 'Prog', 'PSP': 'PSP', 'Resp': 'RSD', 'SD': 'RSD'})
df['label2pr'] = df['label'].map({'Prog': 'Prog', 'PSP': 'RSD', 'Resp': 'RSD', 'SD': 'RSD'})


def _savefig(stem, fname_base):
    for ext in ('png', 'eps', 'tiff'):
        plt.savefig(os.path.join(stem, fname_base + '.' + ext),
                    dpi=300, bbox_inches='tight')


# iRECIST extra-growth histogram for true Progressors
prog_valid = df[(df['label'] == 'Prog') & df['V_next'].notna() & df['V_prog'].notna()].copy()
prog_valid['dV_pct'] = (prog_valid['V_next'] - prog_valid['V_prog']) / prog_valid['V_prog'] * 100.0
lo, hi      = np.percentile(prog_valid['dV_pct'], [2, 95])
plot_vals   = prog_valid.loc[(prog_valid['dV_pct'] >= lo) & (prog_valid['dV_pct'] <= hi), 'dV_pct']

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(plot_vals, bins=50, color='#c0392b', edgecolor='white', linewidth=0.4)
ax.set_xlabel('Volume increase during iRECIST wait period (%)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
_savefig(FUZZ_DIR, 'irecist_extra_growth_prog')
plt.close()
print(f"  Saved: irecist_extra_growth_prog.*  "
      f"(median={prog_valid['dV_pct'].median():.0f}%, n={len(prog_valid)})")


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


def make_rf_pipe():
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
    '3class': (
        ['Prog', 'PSP', 'RSD'], 'label3', None, False, False,
    ),
    '2class_prog_psp': (
        ['Prog', 'PSP'], 'label', ['Prog', 'PSP'], True, False,
    ),
    '2class_prog_rsd': (
        ['Prog', 'RSD'], 'label2pr', None, False, True,
    ),
}
TASK_STEMS = {'3class': '3class', '2class_prog_psp': '2class_prog_psp',
              '2class_prog_rsd': '2class_prog_rsd'}

METHOD_ORDER  = ['RECIST 1.1', 'RF', 'iRECIST']
METHOD_COLORS = {'RECIST 1.1': '#7f8c8d', 'RF': '#e67e22', 'iRECIST': '#2980b9'}
CLASS_COLORS  = {'Prog': '#c0392b', 'PSP': '#2980b9', 'RSD': '#27ae60'}

print("\n" + "=" * 60)
all_results = {}

for task_key, (classes, y_col, label_filter,
               recist_psp_guard, irecist_remap) in TASKS.items():
    print(f"\n>>> Task: {task_key}")
    df_task = df[df['label'].isin(label_filter)].copy() if label_filter else df.copy()
    X = df_task[FEATS]; y = df_task[y_col]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    df_te = df_task.loc[y_te.index]

    recist_pred  = predict_recist(df_te)
    irecist_pred = predict_irecist(df_te, remap_psp=irecist_remap)
    if recist_psp_guard:
        irecist_pred = np.where(irecist_pred == 'RSD', 'Prog', irecist_pred)
    rf_pipe = make_rf_pipe(); rf_pipe.fit(X_tr, y_tr)
    rf_pred = rf_pipe.predict(X_te)

    task_res = {}
    for mname, pred in [('RECIST 1.1', recist_pred), ('RF', rf_pred), ('iRECIST', irecist_pred)]:
        f1_per = f1_score(y_te, pred, labels=classes, average=None, zero_division=0)
        f1_mac = f1_score(y_te, pred, labels=classes, average='macro', zero_division=0)
        kappa  = safe_kappa(y_te, pred)
        bacc   = balanced_accuracy_score(y_te, pred)
        mcc    = matthews_corrcoef(y_te, pred)
        task_res[mname] = {'f1_per': f1_per, 'f1_mac': f1_mac, 'kappa': kappa,
                           'bacc': bacc, 'mcc': mcc}
        cls_str = '  '.join(f'{c}={v:.3f}' for c, v in zip(classes, f1_per))
        print(f"    {mname:12s}  {cls_str}  macro={f1_mac:.3f}  κ={kappa:.3f}  "
              f"BA={bacc:.3f}  MCC={mcc:.3f}")
    all_results[task_key] = (task_res, classes)


def plot_per_class_f1(task_res, classes, stem):
    n_cls = len(classes); x = np.arange(len(METHOD_ORDER)); width = 0.22
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ci, cls in enumerate(classes):
        offset = (ci - (n_cls - 1) / 2) * width
        vals   = [task_res[m]['f1_per'][ci] for m in METHOD_ORDER]
        bars   = ax.bar(x + offset, vals, width, label=cls,
                        color=CLASS_COLORS.get(cls, 'steelblue'), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(METHOD_ORDER, fontsize=11)
    ax.set_ylim(0, 1.18); ax.set_ylabel('F1 score', fontsize=12)
    ax.legend(title='Class', fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); _savefig(FUZZ_DIR, f'f1_{stem}'); plt.close()
    print(f"  Saved: f1_{stem}.*")


def plot_scalar(task_res, metric_key, ylabel, prefix, stem):
    vals   = [task_res[m][metric_key] for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    lo_lim = min(0.0, min(vals) - 0.05)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(METHOD_ORDER, vals, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, vals):
        ypos = bar.get_height() + 0.012 if val >= 0 else bar.get_height() - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)
    ax.set_ylim(lo_lim, 1.15); ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(axis='y', alpha=0.3, lw=0.6)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); _savefig(FUZZ_DIR, f'{prefix}_{stem}'); plt.close()
    print(f"  Saved: {prefix}_{stem}.*")


print("\n=== Generating figures ===")
for task_key, (task_res, classes) in all_results.items():
    stem = TASK_STEMS[task_key]
    plot_per_class_f1(task_res, classes, stem)
    plot_scalar(task_res, 'f1_mac', 'Macro F1',         'macro_f1', stem)
    plot_scalar(task_res, 'kappa',  "Cohen's κ",         'kappa',    stem)
    plot_scalar(task_res, 'bacc',   'Balanced accuracy', 'bacc',     stem)
    plot_scalar(task_res, 'mcc',    'MCC',               'mcc',      stem)

print(f"\nAll figures → {FUZZ_DIR}/")

import ast
import numpy as np
import pandas as pd
from scipy import stats
from agentomics.eval.mining_enhanced_data import build_mdf

# --- Data loading ---
df = pd.read_csv('./paper_tables/agentomics_all.csv')
mdf = build_mdf(df_path='./paper_tables/agentomics_all.csv', mining_path='./paper_tables/code_mined_features.csv')

mdf_agg = mdf.groupby('run_name', as_index=False).agg({
    'used_fm': ['sum', 'mean'],
    'best_iteration_used_fm': ['first'],
    'best_fm_test_mainmetric': ['first'],
    'best_nonfm_test_mainmetric': ['first'],
    'best_fm_val_mainmetric': ['first'],
    'best_nonfm_val_mainmetric': ['first'],
}).reset_index(drop=True)
mdf_agg.columns = ['_'.join(col).strip('_') for col in mdf_agg.columns]
mdf_agg = mdf_agg.merge(df, on='run_name', how='outer')
mdf_agg['best_iteration_used_fm_first'] = mdf_agg['best_iteration_used_fm_first'].fillna(False)
mdf_agg['IsDD'] = mdf_agg['domain'] == 'Drug Discovery'

# --- Outlier scoring ---
AUPRC_RANDOM_BASELINES = {
    'AGO2_CLASH_Hejret2023': 0.5,
    'pkis2-egfr-wt-c-1': 1/5,
    'cyp2d6-substrate-carbonmangels': 0.5,
}
MAE_RANDOM_BASELINES = {
    'lipophilicity-astrazeneca': 0.392 * 2.5,
    'caco2-wang': 0.282 * 2.5,
}
HIGHER_IS_BETTER = {'ACC', 'AUPRC', 'AUROC', 'PEARSON', 'SPEARMAN'}

def get_random_baseline(metric, dataset):
    if metric == 'ACC':   return 0.5
    if metric == 'AUROC': return 0.5
    if metric in ('PEARSON', 'SPEARMAN'): return 0.0
    if metric == 'AUPRC': return AUPRC_RANDOM_BASELINES[dataset]
    if metric == 'MAE':   return MAE_RANDOM_BASELINES[dataset]
    raise ValueError(f'Unknown metric: {metric}')

def add_baseline_score(df):
    df = df.copy()
    df['baseline_score'] = df.apply(lambda r: get_random_baseline(r['main_metric'], r['dataset']), axis=1)
    return df

def add_outlier_scores(df, perf_col='best_iter_test_mainmetric', threshold=0.18, add_sota_to_max=False):
    df = add_baseline_score(df)
    outlier_z = []
    for _, row in df.iterrows():
        metric = row['main_metric']
        dataset = row['dataset']
        value = row[perf_col]
        random_baseline = row['baseline_score']
        dset_vals = df.loc[df['dataset'] == dataset, perf_col].values.tolist()
        if add_sota_to_max:
            dset_vals.append(float(df.loc[df['dataset'] == dataset, 'human_sota'].values[0]))
        dset_vals = np.array(dset_vals)
        if metric in HIGHER_IS_BETTER:
            best = dset_vals.max()
            outlier_z.append((best - value) / (best - random_baseline))
        else:
            best = dset_vals.min()
            outlier_z.append((value - best) / (random_baseline - best))
    df['outlier_z'] = outlier_z
    df['is_outlier'] = df['outlier_z'] > threshold
    return df

mdf_agg = add_outlier_scores(mdf_agg, threshold=0.183, add_sota_to_max=False)

# --- FM normalization (relative to row's own test/val range) ---
def add_norm_metric(row, col_to_normalize, col_to_normalize_by='would_be_test'):
    if col_to_normalize_by not in row or pd.isna(row[col_to_normalize_by]):
        return None
    raw = ast.literal_eval(row[col_to_normalize_by])
    non_nan = np.array([v for v in raw if not pd.isna(v)])
    lower_is_better = row['main_metric'] == 'MAE'
    worst = non_nan.max() if lower_is_better else non_nan.min()
    best  = non_nan.min() if lower_is_better else non_nan.max()
    if best == worst:
        return 1.0
    value = row[col_to_normalize]
    if value is None or pd.isna(value):
        return None
    return (worst - value) / (worst - best) if lower_is_better else (value - worst) / (best - worst)

mdf_agg['best_fm_test_normalized'] = mdf_agg.apply(
    lambda row: add_norm_metric(row, col_to_normalize='best_fm_test_mainmetric_first', col_to_normalize_by='would_be_test'), axis=1)
mdf_agg['best_fm_val_normalized'] = mdf_agg.apply(
    lambda row: add_norm_metric(row, col_to_normalize='best_fm_val_mainmetric_first', col_to_normalize_by='best_val_metric_so_far'), axis=1)

# --- Statistical tests ---
def _is_binary(vals):
    return set(np.unique(vals)).issubset({0, 1, True, False})

def _run_tests(group1, group2):
    if _is_binary(np.concatenate([group1, group2])):
        a = int(group1.sum());  b = int(len(group1) - a)
        c = int(group2.sum());  d = int(len(group2) - c)
        _, p_value = stats.fisher_exact([[a, b], [c, d]])
        print(f'Fisher exact: [[{a},{b}],[{c},{d}]]  PVALUE {p_value:.4f}')
        print(f'Chi squared: {stats.chi2_contingency([[a, b], [c, d]])[1]:.4f}')
    else:
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        print(f'Mann-Whitney U PVALUE {p_value:.4f}')

def check_pvalues(separator, col, df, print_groups=False):
    print(col)
    g1 = df[df[separator] == True][col].dropna().values
    g2 = df[df[separator] == False][col].dropna().values
    stat1, p1 = stats.shapiro(g1)
    stat2, p2 = stats.shapiro(g2)
    print(f"Group 1 (size {len(g1)}): p={p1:.4f}  {'Normal' if p1 > 0.05 else 'Not normal'}")
    print('mean g1', np.mean(g1), 'median', np.median(g1))
    print(f"Group 2 (size {len(g2)}): p={p2:.4f}  {'Normal' if p2 > 0.05 else 'Not normal'}")
    print('mean g2', np.mean(g2), 'median', np.median(g2))
    if print_groups:
        print("Group 1 values:", sorted(g1))
        print("Group 2 values:", sorted(g2))
    stat, p_var = stats.levene(g1, g2)
    print(f"Equal variance: p={p_var:.4f}  {'Yes' if p_var > 0.05 else 'No'}")
    _run_tests(g1, g2)
    print()

# --- P-values ---
check_pvalues(separator='is_outlier', col='val_stealth_mainmetric_corr', df=mdf_agg)
check_pvalues(separator='is_outlier', col='iterations_that_produced_metrics', df=mdf_agg)
check_pvalues(separator='IsDD', col='val_stealth_mainmetric_corr', df=mdf_agg)
check_pvalues(separator='IsDD', col='best_fm_test_normalized', df=mdf_agg, print_groups=False)
check_pvalues(separator='IsDD', col='best_fm_val_normalized', df=mdf_agg, print_groups=False)

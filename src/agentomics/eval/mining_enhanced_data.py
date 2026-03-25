import pandas as pd
import re, ast

def parse_array(s):
    if pd.isna(s):
        return None
    s = re.sub(r'np\.float64\(([^)]+)\)', r'\1', s)  # unwrap np.float64(x) -> x
    s = re.sub(r'\bnan\b', 'None', s)                 # nan -> None for ast
    result = ast.literal_eval(s)
    return [float('nan') if v is None else v for v in result]

def get_at_iter(lookup, run_name, iteration_number):
    arr = lookup.get(run_name)
    if arr is None or iteration_number >= len(arr):
        return None
    v = arr[iteration_number]
    return None if pd.isna(v) else v


def build_mdf(df_path, mining_path):
    df = pd.read_csv(df_path)

    mdf = pd.read_csv(mining_path).replace({'no': False, 'yes': True, 'not_applicable': None})
    ohe_columns_arch = pd.get_dummies(mdf['architecture_category'], prefix='arch')
    ohe_columns_repre = pd.get_dummies(mdf['representation_category'], prefix='repre')
    mdf = pd.concat([mdf, ohe_columns_arch, ohe_columns_repre], axis=1)

    def make_lookup(col):
        return {row['run_name']: parse_array(row[col]) for _, row in df.iterrows()}

    stealth_lookup = make_lookup('stealth_test_per_iter_array')
    stealth_common_lookup = make_lookup('stealth_test_per_iter_array_common')
    val_common_lookup = make_lookup('valid_per_iter_array_common')

    mdf['stealth_test_at_iter'] = mdf.apply(lambda r: get_at_iter(stealth_lookup, r['run_name'], r['iteration_number']), axis=1)
    mdf['stealth_test_common_at_iter'] = mdf.apply(lambda r: get_at_iter(stealth_common_lookup, r['run_name'], r['iteration_number']), axis=1)
    mdf['val_common_at_iter'] = mdf.apply(lambda r: get_at_iter(val_common_lookup, r['run_name'], r['iteration_number']), axis=1)

    mdf['used_fm'] = (
        mdf['arch_transformer_finetune'] |
        mdf['repre_pretrained_embeddings'] |
        mdf['repre_tokenized_sequences_for_finetuning']
    )
    mdf['model_parameter_count_nonfm'] = mdf['model_parameter_count'].where(~mdf['used_fm'])

    import numpy as np
    param_cols = [c for c in mdf.columns if c.startswith('model_parameter_count')]
    for col in param_cols:
        mdf[f'log_{col}'] = np.log(mdf[col])
    mdf.drop(columns=param_cols, inplace=True)

    reg_bool_cols = [c for c in mdf.columns if c.startswith('regularization_') and c != 'regularization_common']
    mdf['regularization_any'] = mdf[reg_bool_cols].any(axis=1)

    # best_iteration_* features: values from the best iteration of each run
    best_iter_map = df.set_index('run_name')['best_iteration_logged'].to_dict()
    best_iter_rows = (
        mdf[mdf.apply(lambda r: r['iteration_number'] == best_iter_map.get(r['run_name']), axis=1)]
        .set_index('run_name')
    )
    best_iter_features = {
        'best_iteration_log_model_parameter_count':      'log_model_parameter_count',
        'best_iteration_log_model_parameter_count_nonfm':'log_model_parameter_count_nonfm',
        'best_iteration_main_metric':                    'main_metric',
        'best_iteration_trained_to_convergence':         'trained_to_convergence',
        'best_iteration_trained_to_early_stopping':      'trained_to_early_stopping',
        'best_iteration_regularization_any':             'regularization_any',
        'best_iteration_used_fm':                        'used_fm',
        'best_iter_stealth_test_at_iter':                'stealth_test_at_iter',
        'best_iter_val_common_at_iter':                  'val_common_at_iter',
    }
    for new_col, src_col in best_iter_features.items():
        mdf[new_col] = mdf['run_name'].map(best_iter_rows[src_col])

    # best stealth test metric split by whether FM was used at that iteration
    HIGHER_IS_BETTER = {'ACC', 'AUPRC', 'AUROC', 'PEARSON', 'SPEARMAN'}
    metric_dir = df.set_index('run_name')['main_metric'].to_dict()

    def best_test_for_fm_flag(run_name, fm_flag, col='stealth_test_at_iter'):
        subset = mdf[(mdf['run_name'] == run_name) & (mdf['used_fm'] == fm_flag)]
        vals = subset[col].dropna()
        if vals.empty:
            return None
        return vals.max() if metric_dir.get(run_name) in HIGHER_IS_BETTER else vals.min()

    run_names = mdf['run_name'].unique()
    mdf['best_fm_test_mainmetric']    = mdf['run_name'].map({r: best_test_for_fm_flag(r, True, 'stealth_test_at_iter')  for r in run_names})
    mdf['best_nonfm_test_mainmetric'] = mdf['run_name'].map({r: best_test_for_fm_flag(r, False, 'stealth_test_at_iter') for r in run_names})

    mdf['best_fm_val_mainmetric'] = mdf['run_name'].map({r: best_test_for_fm_flag(r, True, 'val_common_at_iter')  for r in run_names})
    mdf['best_nonfm_val_mainmetric'] = mdf['run_name'].map({r: best_test_for_fm_flag(r, False, 'val_common_at_iter') for r in run_names})

    return mdf

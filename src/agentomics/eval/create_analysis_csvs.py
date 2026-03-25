import pandas as pd
from generate_method_dfs import generate_zeroshot_runs, generate_iterative_runs

def generate_rg_leaderboard():
    data = {
        "Dataset / Model": [
            "human_enhancers_cohn",
            "human_enhancers_ensembl",
            "human_ocr_ensembl",
            "drosophila_enhancers_stark",
            "AGO2_CLASH_Hejret2023",
        ],
        "FinDNA (finetuned)": [74.20, 93.30, 81.19, None, None],
        "ConvNova": [74.3, 90.0, 79.3, None, None],
        "OmniDNA": [73.8, 91.9, 79.1, None, None],
        "MxDNA": [74.67, 93.13, 81.05, None, None],
        "HyenaDNA": [74.2, 89.2, 80.9, None, None],
        "MSAMamba (model using MSA)": [72.7, 90.1, 82.5, None, None],
        "Janus": [73.4, 86.8, 79.9, None, None],
        "DNABERT-2 (as reported in FinDNA)": [73.1, 92.8, 79.5, None, None],
        "CARMANIA": [72.5, 89.5, 77.8, None, None],
        "KAN": [74.0, 80.2, 78.4, 58.4, None],
        "SwanDNA": [73.97, 90.32, 77.52, None, None],
        "Zurgham (best reported)": [71.8, 82.1, None, 68.6, None],
        "mirbench": [None, None, None, None, 86.0],
    }

    df = pd.DataFrame(data)
    df.to_csv('paper_tables/regulatory_genomics_leaderboard.csv', index=False)
    return df

def get_gb_leaderboard():
    gb_leaderboard_csv = pd.read_csv('paper_tables/regulatory_genomics_leaderboard.csv')
    gb_leaderboard = {
        row["Dataset / Model"]: [round(v,3) for v in (row.drop("Dataset / Model").dropna() / 100).tolist()]
        for _, row in gb_leaderboard_csv.iterrows()
    }
    return gb_leaderboard

def generate_dataset_sizes():
    datasize_data = [
        {'dataset':'adme-fang-SOLU-1', 'train_size': 1578, 'test_size':400},
        {'dataset':'pkis2-egfr-wt-c-1', 'train_size': 496, 'test_size': 144},
        {'dataset':'caco2-wang', 'train_size':728, 'test_size':182},
        {'dataset':'adme-fang-HCLint-1', 'train_size':2229, 'test_size':575},
        {'dataset':'adme-fang-HPPB-1', 'train_size':126, 'test_size':34},
        {'dataset':'bbb-martins', 'train_size':1624, 'test_size':406},
        {'dataset':'herg', 'train_size':523, 'test_size':132},
        {'dataset':'cyp2d6-substrate-carbonmangels', 'train_size':532, 'test_size':135},
        {'dataset':'lipophilicity-astrazeneca', 'train_size':3360, 'test_size':840},

        {'dataset':'Proteingym-DMS dataset: CBX4_HUMAN_Tsuboyama_2023_2K28', 'train_size':918, 'test_size':None},
        {'dataset':'Proteingym-DMS dataset: CSN4_MOUSE_Tsuboyama_2023_1UFM_indels', 'train_size':197, 'test_size':None},
        {'dataset':'Proteingym-DMS dataset: PSAE_PICP2_Tsuboyama_2023_1PSE_indels', 'train_size':1220, 'test_size':None},
        {'dataset':'Proteingym-DMS dataset: Q8EG35_SHEON_Campbell_2022_indels', 'train_size':332, 'test_size':None},
        {'dataset':'Proteingym-DMS dataset: SPA_STAAU_Tsuboyama_2023_1LP1', 'train_size':1036, 'test_size':None},

        {'dataset':'Proteingym-DMS dataset: SPIKE_SARS2_Starr_2020_binding', 'train_size':3803, 'test_size':None},
        {'dataset':'AGO2_CLASH_Hejret2023', 'train_size':8193, 'test_size':965},
        {'dataset':'drosophila_enhancers_stark', 'train_size':5184, 'test_size':1730},
        {'dataset':'human_enhancers_cohn', 'train_size':20843, 'test_size':6948},
        {'dataset':'human_enhancers_ensembl', 'train_size':123872, 'test_size':30970},
        {'dataset':'human_ocr_ensembl', 'train_size':139804, 'test_size':34952},
    ]
    train_datasize_dict = {dataset_dict['dataset']:dataset_dict['train_size'] for dataset_dict in datasize_data}
    test_datasize_dict = {dataset_dict['dataset']:dataset_dict['test_size'] for dataset_dict in datasize_data}
    return train_datasize_dict, test_datasize_dict

def get_dset_to_mainmetric():
    return {
        'AGO2_CLASH_Hejret2023':'AUPRC',
        'drosophila_enhancers_stark': 'ACC',
        'human_enhancers_cohn': 'ACC',
        'human_enhancers_ensembl': 'ACC',
        'human_ocr_ensembl': 'ACC',

        'polarishub/polaris-pkis2-egfr-wt-c-1': 'AUPRC',
        'polarishub/polaris-adme-fang-hclint-1': 'PEARSON',
        'polarishub/polaris-adme-fang-solu-1': 'PEARSON',
        'polarishub/tdcommons-lipophilicity-astrazeneca': 'MAE',
        'polarishub/tdcommons-cyp2d6-substrate-carbonmangels': 'AUPRC',
        'polarishub/polaris-adme-fang-hppb-1': 'PEARSON',
        'polarishub/tdcommons-herg': 'AUROC',
        'polarishub/tdcommons-bbb-martins': 'AUROC',
        'polarishub/tdcommons-caco2-wang': 'MAE',

        'proteingym-dms/SPIKE_SARS2_Starr_2020_binding': 'SPEARMAN',
        'proteingym-dms/SPA_STAAU_Tsuboyama_2023_1LP1': 'SPEARMAN',
        'proteingym-dms/PSAE_PICP2_Tsuboyama_2023_1PSE_indels': 'SPEARMAN',
        'proteingym-dms/CBX4_HUMAN_Tsuboyama_2023_2K28': 'SPEARMAN',
        'proteingym-dms/Q8EG35_SHEON_Campbell_2022_indels': 'SPEARMAN',
        'proteingym-dms/CSN4_MOUSE_Tsuboyama_2023_1UFM_indels': 'SPEARMAN',

    }

def get_dset_domain(dset):
    if 'proteingym' in dset:
        return 'Protein Engineering'
    if 'polarishub' in dset:
        return 'Drug Discovery'
    return 'Regulatory Genomics'


def generate_zeroshot_summaries(zeroshot_df: pd.DataFrame, dset_to_metric, gb_leaderboard, file_name='zeroshot_stats.csv'):
    import numpy as np
    import pandas as pd

    gb_mask = zeroshot_df['dataset'].isin(gb_leaderboard.keys())
    zeroshot_df.loc[gb_mask, 'leaderboard%'] = zeroshot_df[gb_mask].apply(
        lambda x: 100*sum(x[dset_to_metric[x['dataset']]] > np.array(gb_leaderboard[x['dataset']]))/len(gb_leaderboard[x['dataset']]) if not pd.isna(x[dset_to_metric[x['dataset']]]) else None, axis=1)

    for column in ['biomlbench/leaderboard_percentile', 'MAE', 'PEARSON', 'SPEARMAN', 'ACC', 'AUROC', 'AUPRC']:
        # if col doesnt exist, add an empty one
        if column not in zeroshot_df.columns:
            zeroshot_df[column] = None

    zeroshot_df.loc[~gb_mask, 'leaderboard%'] = zeroshot_df.loc[~gb_mask, 'biomlbench/leaderboard_percentile']
    zeroshot_df['domain'] = zeroshot_df.apply(lambda x: get_dset_domain(x['dataset']), axis=1)

    zeroshot_summary = (
        zeroshot_df
        .groupby('dataset')
        .agg(
            successful_run_mean=('successful_run', 'mean'),

            acc_max=('ACC','max'),
            auprc_max=('AUPRC','max'),
            auroc_max=('AUROC', 'max'),
            mae_min=('MAE','min'),
            pearson_max=('PEARSON', 'max'),
            spearman_max=('SPEARMAN', 'max'),

            leaderboard_perc_mean=('leaderboard%', 'mean'),
            leaderboard_perc_std=('leaderboard%', 'std'),
            leaderboard_perc_sem=('leaderboard%', 'sem'),

        )
        .reset_index()
    )
    zeroshot_summary_domains = (
        zeroshot_df
        .groupby('domain')
        .agg(
            successful_run_mean=('successful_run', 'mean'),

            acc_max=('ACC','max'),
            auprc_max=('AUPRC','max'),
            auroc_max=('AUROC', 'max'),
            mae_min=('MAE','min'),
            pearson_max=('PEARSON', 'max'),
            spearman_max=('SPEARMAN', 'max'),

            leaderboard_perc_mean=('leaderboard%', 'mean'),
            leaderboard_perc_std=('leaderboard%', 'std'),
            leaderboard_perc_sem=('leaderboard%', 'sem'),


        )
    ).reset_index()
    zeroshot_summary_domains['dataset'] = zeroshot_summary_domains['domain']
    zeroshot_summary_domains = zeroshot_summary_domains.drop(columns=['domain'])

    zeroshot_summary_all = zeroshot_df.groupby(lambda _: 0).agg(
            successful_run_mean=('successful_run', 'mean'),

            acc_max=('ACC','max'),
            auprc_max=('AUPRC','max'),
            auroc_max=('AUROC', 'max'),
            mae_min=('MAE','min'),
            pearson_max=('PEARSON', 'max'),
            spearman_max=('SPEARMAN', 'max'),

            leaderboard_perc_mean=('leaderboard%', 'mean'),
            leaderboard_perc_std=('leaderboard%', 'std'),
            leaderboard_perc_sem=('leaderboard%', 'sem'),


        )
    
    zeroshot_summary_all['dataset'] = ['All Domains']
    zeroshot_info_df = pd.concat([zeroshot_summary,zeroshot_summary_all,zeroshot_summary_domains])
    zeroshot_info_df.to_csv(f'./paper_tables/{file_name}')


def generate_arch_csv(fin_df):
    arch_df = pd.read_csv('./paper_tables/best_runs_architecture_table_final.csv')
    best_runs_names = [
        'ecstatic_movie_bumped', 
        'impatient_tractor_race',
        'funny_computing_spot', 
        'traveled_cabal_teased',
        'listening_weasel_collect',
        'outraged_boomer_dried',
        'replenished_brightness_snatched',
        'corporatist_melodrama_admitted',
        'operating_beaver_strap',
        'grumpy_cougar_bare',
        'unjust_hoarding_program',
        'swooning_sweater_picked',
        'branched_whisky_shopped',
        'deodorant_metre_push',
        'conditioned_mink_close',
        'fact_colors_preach',
        'autobiographical_frog_supposed',
        'dismal_mechanics_cycle',
        'tenacious_ewe_appear',
        'robust_ralph_moan',
    ]
    classes_map = {
        "cnn": "CNN",
        "mlp": "MLP",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "svm": "SVM",
        "logistic_regression": "Log. Regression",
        "linear_regression": "Lin. Regression",
        "naive_bayes": "Naive Bayes",
        "knn": "k-NN",
        "ensemble": "Ensemble",
        "transformer_trained": "Transformer",
        "foundation_model_embeddings_frozen": "FM Embeddings",
        "foundation_model_trained": "FM Fine-Tuning",
        # "no_match": "No Match",
        "is_hybrid_model": "Hybrid Model",
    }
    simple_archs = [
        "CNN",
        "MLP",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
        "SVM",
        "Log. Regression",
        "Lin. Regression",
        "Naive Bayes",
        "k-NN",
        "Transformer",
        "FM Embeddings",
        "FM Fine-Tuning",
        # "ensemble": "Ensemble",
    ]
    arch_cols = list(classes_map.keys())
    f_arch_df = arch_df[arch_df['run_name'].isin(best_runs_names)]
    f_arch_df['ensemble'] = f_arch_df.apply(lambda row: True if row['random_forest'] == True else row['ensemble'], axis=1)
    f_arch_df['ensemble'] = f_arch_df.apply(lambda row: True if row['xgboost'] == True else row['ensemble'], axis=1)
    arch_df_extra = fin_df.merge(f_arch_df, on='run_name')

    # add mis-logged runs manually
    for run_name, arch, iter in [('funny_computing_spot', 'ensemble', 33), ('robust_ralph_moan','foundation_model_embeddings_frozen',5),  ('robust_ralph_moan','logistic_regression',5), ('swooning_sweater_picked', 'foundation_model_trained', 5), ('swooning_sweater_picked', 'linear_regression', 5)]:
        arch_df_extra.loc[arch_df_extra['run_name'] == run_name,'no_match'] = False
        arch_df_extra.loc[(arch_df_extra['run_name'] == run_name) & (iter == arch_df_extra['iteration_number']) ,arch] = True 
        mask = (
            (arch_df_extra['run_name'] == run_name) &
            (arch_df_extra['iteration_number'] == iter)
        )

        if not mask.any() and run_name == 'funny_computing_spot':
            # create missing row
            new_row = {
                'run_name': run_name,
                'iteration_number': iter,
                'arch': arch,
                arch: True,
                'best_iteration_logged': iter,
                'foundation_model_embeddings_frozen': False,
            }
            arch_df_extra.loc[len(arch_df_extra)] = new_row

    arch_df_extra = arch_df_extra.drop(columns=['no_match'])
    arch_df_extra['arch'] = arch_df_extra[arch_cols].apply(
        lambda row: '/'.join([classes_map[col] for col in arch_cols if row[col]]),
        axis=1
    )

    arch_df_extra['og_arch'] = arch_df_extra.apply(lambda row: row['arch'], axis=1)
    arch_df_extra['arch'] = arch_df_extra.apply(lambda row: 'Custom Ensemble' if row['ensemble'] == True else row['arch'], axis=1)
    arch_df_extra['arch'] = arch_df_extra.apply(lambda row: 'XGBoost' if ((row['ensemble'] == True) & (row['xgboost'] == True)) else row['arch'], axis=1)
    arch_df_extra['arch'] = arch_df_extra.apply(lambda row: 'Random Forest' if ((row['ensemble'] == True) & (row['random_forest'] == True)) else row['arch'], axis=1)
    arch_df_extra['arch'] = arch_df_extra.apply(lambda row: 'Hybrid Model' if (row['is_hybrid_model'] == True) & (row['ensemble'] == False) else row['arch'], axis=1)

    special_arch_map = {
        'XGBoost/FM Embeddings': 'XGBoost',
        'SVM/FM Embeddings': 'SVM',
        'Transformer/FM Fine-Tuning': 'FM Fine-Tuning',
        'MLP/FM Embeddings': 'MLP',
        'Lin. Regression/FM Embeddings': 'Lin. Regression',
        'Lin. Regression/FM Fine-Tuning': 'FM Fine-Tuning',
        'SVM/FM Embeddings/FM Fine-Tuning': 'Hybrid Model',
        'Log. Regression/FM Embeddings': 'Log. Regression',
        'FM Embeddings/FM Fine-Tuning': 'FM Fine-Tuning',
    }
    arch_df_extra['arch'] = arch_df_extra.apply(lambda row: special_arch_map[row['arch']] if row['arch'] in special_arch_map.keys() else row['arch'] , axis=1)
    arch_df_extra['arch_type'] = arch_df_extra.apply(lambda row: 'simple' if row['arch'] in simple_archs else 'complex', axis=1)

    arch_df_extra[arch_df_extra['arch_type']=='simple']['arch'].value_counts()
    arch_df_extra[arch_df_extra['arch_type']=='complex']['arch'].value_counts()
    arch_df_extra = arch_df_extra[(arch_df_extra['arch'] != '') & (arch_df_extra['arch'] != 'FM Embeddings')]
    arch_df_extra.to_csv('./paper_tables/architectures.csv' ,index=False)
    return arch_df_extra


def create_group_summaries(fin_df, suffix=''):
    all_summary = (
        fin_df
        .groupby(lambda _: 0)
        .agg(
            best_iter_test_mainmetric_mean=('best_iter_test_mainmetric', 'mean'),
            leaderboard_mean=('leaderboard%', 'mean'),
            leaderboard_std=('leaderboard%', 'std'),
            leaderboard_var=('leaderboard%', 'var'),
            leaderboard_sem=('leaderboard%', 'sem'),

            run_name_count=('run_name', 'count'),
            is_new_sota_sum=('is_new_sota', 'sum'),
            usd_cost_mean=('usd_cost', 'mean'),
            cost_per_iter_mean=('cost_per_iter', 'mean'),
            iterations_completed_mean=('iterations_completed', 'mean'),
            iterations_that_produced_metrics_mean=('iterations_that_produced_metrics', 'mean'),
            iterations_that_produced_metrics_std=('iterations_that_produced_metrics', 'std'),
            iterations_perc_that_produced_metrics_mean=('iteration%_produced_metrics', 'mean'),
            iterations_perc_that_produced_metrics_std=('iteration%_produced_metrics', 'std'),
            number_of_trainval_splits_mean=('number_of_trainval_splits', 'mean'),
            best_iteration_logged_mean=('best_iteration_logged', 'mean'),
            successful_run_sum=('successful_run', 'sum'),
            mean_splits=('number_of_trainval_splits', 'mean'),
        )
        .reset_index(drop=True)
    )   
    per_dataset_summary = (
        fin_df
        .groupby('dataset')
        .agg(
            best_iter_test_mainmetric_mean=('best_iter_test_mainmetric', 'mean'),
            best_iter_test_mainmetric_max=('best_iter_test_mainmetric', 'max'),
            best_iter_test_mainmetric_min=('best_iter_test_mainmetric', 'min'),
            best_iter_test_mainmetric_std=('best_iter_test_mainmetric', 'std'),
            best_iter_test_mainmetric_var=('best_iter_test_mainmetric', 'var'),
            human_sota_max=('human_sota', 'max'),
            leaderboard_mean=('leaderboard%', 'mean'),
            leaderboard_std=('leaderboard%', 'std'),
            leaderboard_var=('leaderboard%', 'var'),
            leaderboard_sem=('leaderboard%', 'sem'),
            run_name_count=('run_name', 'count'),
            is_new_sota_sum=('is_new_sota', 'sum'),
            usd_cost_mean=('usd_cost', 'mean'),
            cost_per_iter_mean=('cost_per_iter', 'mean'),
            iterations_completed_mean=('iterations_completed', 'mean'),
            iterations_that_produced_metrics_mean=('iterations_that_produced_metrics', 'mean'),
            iterations_that_produced_metrics_std=('iterations_that_produced_metrics', 'std'),
            iterations_perc_that_produced_metrics_mean=('iteration%_produced_metrics', 'mean'),
            iterations_perc_that_produced_metrics_std=('iteration%_produced_metrics', 'std'),            
            number_of_trainval_splits_mean=('number_of_trainval_splits', 'mean'),
            best_iteration_logged_mean=('best_iteration_logged', 'mean'),
            successful_run_sum=('successful_run', 'sum'),
            mean_splits=('number_of_trainval_splits', 'mean'),
        )
        .reset_index()
    )
    per_dataset_type_summary = (
        fin_df
        .groupby('domain')
        .agg(
            best_iter_test_mainmetric_mean=('best_iter_test_mainmetric', 'mean'),
            leaderboard_mean=('leaderboard%', 'mean'),
            leaderboard_std=('leaderboard%', 'std'),
            leaderboard_var=('leaderboard%', 'var'),
            leaderboard_sem=('leaderboard%', 'sem'),
            run_name_count=('run_name', 'count'),
            is_new_sota_sum=('is_new_sota', 'sum'),
            usd_cost_mean=('usd_cost', 'mean'),
            cost_per_iter_mean=('cost_per_iter', 'mean'),
            iterations_completed_mean=('iterations_completed', 'mean'),
            iterations_that_produced_metrics_mean=('iterations_that_produced_metrics', 'mean'),
            iterations_that_produced_metrics_std=('iterations_that_produced_metrics', 'std'),
            iterations_perc_that_produced_metrics_mean=('iteration%_produced_metrics', 'mean'),
            iterations_perc_that_produced_metrics_std=('iteration%_produced_metrics', 'std'),            
            number_of_trainval_splits_mean=('number_of_trainval_splits', 'mean'),
            best_iteration_logged_mean=('best_iteration_logged', 'mean'),
            successful_run_sum=('successful_run', 'sum'),
            mean_splits=('number_of_trainval_splits', 'mean'),
        )
        .reset_index()
    )
    per_dataset_summary.to_csv(f'./paper_tables/agg_per_dataset_agentomics{suffix}.csv')
    per_dataset_type_summary.to_csv(f'./paper_tables/agg_per_domain_agentomics{suffix}.csv')
    all_summary.to_csv(f'./paper_tables/agg_all_agentomics{suffix}.csv')


def generate_corr_tables(fin_df):
    st_fin_df = fin_df[~fin_df['would_be_test'].isna()] #only exps that have stealth test

    def compute_norm_testmetric_achieved(row):
        sub_df = st_fin_df[st_fin_df['run_name'] == row['run_name']]
        if(len(sub_df) == 0):
            return None
        assert len(sub_df) == 1
        iter_to_nonnan_wouldbetest = {}
        assert len(sub_df[0:1]['would_be_test'].values[0]) == sub_df['iterations_completed'].values[0], f"{len(sub_df[0:1]['would_be_test'].values[0])} != {sub_df['iterations_completed'].values[0], sub_df['run_name']}"
        for i, would_be_test_value in enumerate(sub_df[0:1]['would_be_test'].values[0]):
            if not pd.isna(would_be_test_value):
                iter_to_nonnan_wouldbetest[i] = would_be_test_value

        lower_is_better = row['main_metric'] in ['MAE', 'RMSE', 'LOG_LOSS']
        worst_test_metric = max(iter_to_nonnan_wouldbetest.values()) if lower_is_better else min(iter_to_nonnan_wouldbetest.values())
        best_test_metric  = min(iter_to_nonnan_wouldbetest.values()) if lower_is_better else max(iter_to_nonnan_wouldbetest.values())
        best_iter = int(sub_df[0:1]['best_iteration_logged'].values[0])
        if best_iter not in iter_to_nonnan_wouldbetest.keys():
            print('warning', sub_df[0:1]['run_name'].values[0], 'missing best iter test value')
            return None
        best_iter_wouldbetest = iter_to_nonnan_wouldbetest[best_iter]
        if(best_test_metric - worst_test_metric) == 0:
            return 1
        if lower_is_better:
            normalized_best_iter_wouldbetest = (worst_test_metric - best_iter_wouldbetest) / (worst_test_metric - best_test_metric)
        else:
            normalized_best_iter_wouldbetest = (best_iter_wouldbetest - worst_test_metric) / (best_test_metric - worst_test_metric)
        return normalized_best_iter_wouldbetest
    def compute_correlation_table(df):
        import pandas as pd
        
        # Define correlation pairs
        sub_df_with_testval_corr = df[~df['would_be_test'].isna()] #only exps that have stealth test
        print('full stats from', len(df), 'runs')
        print('st stats from', len(sub_df_with_testval_corr), 'runs')

        corr_pairs = [
            ('log_train_size', 'leaderboard%'),
            ('log_test_size', 'leaderboard%'),
            ('iterations_completed', 'leaderboard%'),
            ('iterations_that_produced_metrics', 'leaderboard%'),
            ('iteration%_produced_metrics', 'leaderboard%'),
        ]
        corr_pairs_st = [
            ('log_train_size', 'val_stealth_mainmetric_corr'),
            ('log_test_size', 'val_stealth_mainmetric_corr'),
            ('val_stealth_mainmetric_corr', 'leaderboard%'),
            ('norm_testmetric_achieved', 'val_stealth_mainmetric_corr'),
            ('norm_testmetric_achieved', 'log_train_size'),
            ('norm_testmetric_achieved', 'leaderboard%'),
        ]
        
        # Compute overall correlations
        results = {f"{x} × {y} pearson": df[[x, y]].corr(method='pearson').iloc[0, 1] for x, y in corr_pairs}
        results = {**results, **{f"{x} × {y} spearman": df[[x, y]].corr(method='spearman').iloc[0, 1] for x, y in corr_pairs}}
        results_st = {f"{x} × {y} pearson": sub_df_with_testval_corr[[x, y]].corr(method='pearson').iloc[0, 1] for x, y in corr_pairs_st}
        results_st = {**results_st, **{f"{x} × {y} spearman": sub_df_with_testval_corr[[x, y]].corr(method='spearman').iloc[0, 1] for x, y in corr_pairs_st}}
        for k,v in results_st.items():
            results[k] = v

        for col in ['best_iteration_at_%_of_total','best_iteration_at_h', 'leaderboard%', 'val_stealth_mainmetric_corr', 'cost_per_iter', 'duration_per_iter', 'iterations_completed', 'usd_cost']:
            results[col] = f"{df[col].mean():.3f} ± {df[col].std():.3f}"
        results['val_stealth_mainmetric_corr_median'] = f"{sub_df_with_testval_corr['val_stealth_mainmetric_corr'].median():.3f}"
        
        return pd.Series(results)
    
    fin_df['norm_testmetric_achieved'] = fin_df.apply(lambda row: compute_norm_testmetric_achieved(row), axis=1)
    all_corrs = compute_correlation_table(fin_df)
    all_corrs.to_csv('./paper_tables/all_misc_stats.csv')
    for dataset_type in fin_df['domain'].unique():
        sub_df = fin_df[fin_df['domain'] == dataset_type]
        corrs_df = compute_correlation_table(sub_df)
        corrs_df.to_csv(f'./paper_tables/{dataset_type}_misc_stats.csv')   

def main():
    generate_rg_leaderboard()
    zeroshot_df = generate_zeroshot_runs(tags=["ismb2026_zeroshot_v1", "imsb_2026_zeroshot_v2"], path='./paper_tables/zeroshots.csv')
    dataset_to_metric = get_dset_to_mainmetric()
    generate_zeroshot_summaries(zeroshot_df, dataset_to_metric, get_gb_leaderboard(), file_name='zeroshot_stats.csv')
    train_datasize_dict, test_datasize_dict = generate_dataset_sizes()
    
    agentomics_all_df = generate_iterative_runs(tags=[
        'ismb2026_v2', 
        'ismb2026_v3', 
        'ismb2026_v4',
        'ismb2026_v5',
        'ismb2026_v6',
        'ismb2026_v7',
        ], 
        train_datasize_dict=train_datasize_dict,
        test_datasize_dict=test_datasize_dict,
        path='./paper_tables/agentomics_all.csv', 
        gb_leaderboard=get_gb_leaderboard()
    )

    generate_arch_csv(agentomics_all_df)
    create_group_summaries(agentomics_all_df)
    generate_corr_tables(agentomics_all_df)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

def analyse_df(df):
    # Only successful runs
    successes_df = df[df['successful_run'] == 1].groupby(['dataset', 'model', 'method']).agg({
        'AUPRC': ['mean', 'max', 'std'],
        'AUROC': ['mean', 'max', 'std'],
        'ACC': ['mean', 'max', 'std'],
        'successful_run': ['mean'],
        'duration': ['mean'],
        'run_name': ['nunique'],
    }).reset_index()
    # All runs
    all_df = df.groupby(['dataset', 'model', 'method']).agg({
        'successful_run': ['mean'],
        'run_name': ['nunique'],
    }).reset_index()

    complete_df = successes_df.merge(all_df, on=['dataset', 'model', 'method'], suffixes=('_nonfails', ''), how='outer')
    complete_df.columns = complete_df.columns.to_flat_index().map('_'.join).str.strip('_')
    return complete_df


def squash_max_rows(df, prefix, new_col_name):
        methods = df.index[df.index.str.startswith(prefix)]
        methods_df = df.loc[methods]
        methods_df = methods_df.max(axis=0).to_frame().T
        methods_df.index = [new_col_name]
        df = df.drop(methods)
        df = pd.concat([df, methods_df], axis=0, join='outer', ignore_index=False)
        return df

def squash_mean_rows(df, prefix, new_col_name):
        methods = df.index[df.index.str.startswith(prefix)]
        methods_df = df.loc[methods]
        methods_df = methods_df.mean(axis=0).to_frame().T
        methods_df.index = [new_col_name]
        df = df.drop(methods)
        df = pd.concat([df, methods_df], axis=0, join='outer', ignore_index=False)
        return df

def feedback_analysis():
    agentomics_df = pd.read_csv('FINAL_TABLES/Agentomics_runs.csv')
    count_iteration_is_better = 0
    for row in agentomics_df.iterrows():
        if row[1]['dataset'] == 'AGO2_CLASH_Hejret2023':
            metric = 'AUPRC'
        else:
            metric = 'ACC'
        if (row[1][f"{metric}_zeroth"] <= row[1][metric]):
            count_iteration_is_better += 1

    print('% of runs that were better with feedback than without', count_iteration_is_better/len(agentomics_df))
    
    df = pd.read_csv('FINAL_TABLES/max_metrics_and_sota_df.csv')
    def get_row_where_method(df, method):
        return df[df['method'] == method].iloc[0]
    row_ag = get_row_where_method(df, 'Agentomics:GPT4_1')
    row_ag_no_feedback = get_row_where_method(df, 'Agentomics:GPT4_1_no_feedback')

    rel_diff = (row_ag.values[1:] - row_ag_no_feedback.values[1:])/ row_ag_no_feedback.values[1:]
    abs_diff = row_ag.values[1:] - row_ag_no_feedback.values[1:]

    print('RELATIVE DIFFERENCE', np.mean(rel_diff))
    print('ABSOLUTE DIFFERENCE', np.mean(abs_diff))

if __name__ == "__main__":
    path_prefix = 'FINAL_TABLES/'
    zeroshot_df = pd.read_csv(path_prefix + "zeroshot_runs.csv")
    di_df = pd.read_csv(path_prefix + "DI_runs.csv")
    agentomics_df = pd.read_csv(path_prefix + "Agentomics_runs.csv")
    aide_df = pd.read_csv(path_prefix + "AIDE_runs.csv")
    sota_df = pd.read_csv(path_prefix + "SOTA_leaderboard.csv")

    def map_to_no_feedback_runs(df):
        new_rows = []
        for _, row in df.iterrows():
            new_rows.append({
                'dataset': row['dataset'],
                'model': row['model'],
                'method': row['method'] + "_no_feedback",
                'successful_run': row['successful_run'],
                'run_name': row['run_name'] + "_no_feedback",
                'AUPRC': row['0:stealth_test/AUPRC'],
                'AUROC': row['0:stealth_test/AUROC'],
                'ACC': row['0:stealth_test/ACC'],
                'duration': -1,
            })
        return pd.DataFrame(new_rows)
    
    # Compute p-values for Agentomics feedback vs no feedback version
    p_values = {}
    agentomics_no_feedback_df = map_to_no_feedback_runs(agentomics_df)
    all_agentomics = pd.concat([agentomics_no_feedback_df, agentomics_df], axis=0, join='outer', ignore_index=True)
    for dataset in all_agentomics['dataset'].unique():
         # for each dataset, compute difference between the two methods
        if(dataset == "AGO2_CLASH_Hejret2023"):
            metric = "AUPRC"
        else:
            metric = "ACC"
        df = all_agentomics[all_agentomics['dataset'] == dataset]
        acc_no_feedback = df[df['method'] == 'Agentomics:GPT4_1_no_feedback'][metric].values
        acc_feedback = df[df['method'] == 'Agentomics:GPT4_1'][metric].values
        stats = ttest_rel(acc_no_feedback, acc_feedback)
        p_values[dataset] = stats.pvalue
    p_values_df = pd.DataFrame(p_values.items(), columns=['dataset', 'p_value']) 
    p_values_df.to_csv(path_prefix + "p_values_feedback_vs_nofeedback.csv", index=False)

    # Aggregate all replicate runs
    zeroshot_agg = analyse_df(zeroshot_df)
    di_agg = analyse_df(di_df)
    agentomics_agg = analyse_df(agentomics_df)
    agentomics_no_feedback_agg = analyse_df(agentomics_no_feedback_df)
    aide_agg = analyse_df(aide_df)
    exp_dfs = [zeroshot_agg, di_agg, agentomics_agg, aide_agg, agentomics_no_feedback_agg]

    experiments_df = pd.concat(exp_dfs, axis=0, join='outer', ignore_index=True)
    experiments_df.to_csv(path_prefix + "all_aggregated_experiments.csv", index=False)

    # MAX metrics and SOTA
    experiments_acc_max = experiments_df.pivot(columns="dataset", values="ACC_max", index="method")
    experiments_auprc_max = experiments_df.pivot(columns="dataset", values="AUPRC_max", index="method")

    experiments_acc_max = experiments_acc_max.drop(columns=["AGO2_CLASH_Hejret2023"])
    max_metrics_df = pd.concat([experiments_acc_max, experiments_auprc_max['AGO2_CLASH_Hejret2023']], axis=1)

    sota_df['metric'] = sota_df['ACC_max'].fillna(sota_df['AUPRC_max'])
    sota_df['method'] = 'Human SOTA'
    sota_df_agg = sota_df.pivot(columns="dataset", values="metric", index="method")

    max_and_sota_df = pd.concat([max_metrics_df, sota_df_agg], axis=0, join='outer', ignore_index=False)
    max_and_sota_df.to_csv(path_prefix + "max_metrics_and_sota_df.csv", index=True)

    # take all methods that start with DI or zero_shot and make them into one row with max of all
    max_and_sota_df_small = squash_max_rows(max_and_sota_df, "DI", "DI (max of all models)")
    max_and_sota_df_small = squash_max_rows(max_and_sota_df_small, "zero_shot", "zero_shot (max of all models)")

    max_and_sota_df_small.to_csv(path_prefix + "max_metrics_and_sota_df_small.csv", index=True) 
    

    # MEAN metrics
    experiments_acc_mean = experiments_df.pivot(columns="dataset", values="ACC_mean", index="method")
    experiments_auprc_mean = experiments_df.pivot(columns="dataset", values="AUPRC_mean", index="method")
    experiments_acc_mean = experiments_acc_mean.drop(columns=["AGO2_CLASH_Hejret2023"])
    mean_metrics_df = pd.concat([experiments_acc_mean, experiments_auprc_mean['AGO2_CLASH_Hejret2023']], axis=1)
    mean_metrics_df.to_csv(path_prefix + "mean_metrics_df.csv", index=True)

    # STD metrics
    experiments_acc_std = experiments_df.pivot(columns="dataset", values="ACC_std", index="method")
    experiments_auprc_std = experiments_df.pivot(columns="dataset", values="AUPRC_std", index="method")
    experiments_acc_std = experiments_acc_std.drop(columns=["AGO2_CLASH_Hejret2023"])
    std_metrics_df = pd.concat([experiments_acc_std, experiments_auprc_std['AGO2_CLASH_Hejret2023']], axis=1)
    std_metrics_df.to_csv(path_prefix + "std_metrics_df.csv", index=True)
    
    # SUCCESS RATES
    experiments_succ_rates = experiments_df.pivot(columns="dataset", values="successful_run_mean", index="method")
    experiments_succ_rates.to_csv(path_prefix + "experiments_succ_rates.csv", index=True)

    succ_df_small = squash_mean_rows(experiments_succ_rates, "DI", "DI (mean of all models)")
    succ_df_small = squash_mean_rows(succ_df_small, "zero_shot", "zero_shot (mean of all models)")

    succ_df_small = succ_df_small.mean(axis=1).to_frame('all_datasets_mean')
    succ_df_small.to_csv(path_prefix + "experiments_succ_rates_small.csv", index=True)

    # MEAN duration
    experiments_duration_mean = experiments_df.pivot(columns="dataset", values="duration_mean", index="method")
    experiments_duration_mean.to_csv(path_prefix + "experiments_duration_mean.csv", index=True)

    duration_df_small = squash_mean_rows(experiments_duration_mean, "DI", "DI (mean of all models)")
    duration_df_small = squash_mean_rows(duration_df_small, "zero_shot", "zero_shot (mean of all models)")
    duration_df_small.to_csv(path_prefix + "experiments_duration_mean_small.csv", index=True)

    # Feedback vs no-feedback agentomics analysis
    feedback_analysis()

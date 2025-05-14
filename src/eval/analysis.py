import pandas as pd

def analyse_df(df):
    # Only successful runs
    successes_df = df[df['successful_run'] == 1].groupby(['dataset', 'model', 'method']).agg({
        'AUPRC': ['mean', 'max'],
        'AUROC': ['mean', 'max'],
        'ACC': ['mean', 'max'],
        'successful_run': ['mean'],
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


if __name__ == "__main__":
    zeroshot_df = pd.read_csv("zeroshot_runs.csv")
    di_df = pd.read_csv("DI_runs.csv")
    agentomics_df = pd.read_csv("Agentomics_runs.csv")
    sota_df = pd.read_csv("SOTA_leaderboard.csv")

    zeroshot_agg = analyse_df(zeroshot_df)
    di_agg = analyse_df(di_df)
    agentomics_agg = analyse_df(agentomics_df)
    exp_dfs = [zeroshot_agg, di_agg, agentomics_agg]

    experiments_df = pd.concat(exp_dfs, axis=0, join='outer', ignore_index=True)
    experiments_df.to_csv("experiments.csv", index=False)

    # MAX metrics and SOTA
    experiments_acc_max = experiments_df.pivot(columns="dataset", values="ACC_max", index="method")
    experiments_auprc_max = experiments_df.pivot(columns="dataset", values="AUPRC_max", index="method")

    experiments_acc_max = experiments_acc_max.drop(columns=["AGO2_CLASH_Hejret2023"])
    max_metrics_df = pd.concat([experiments_acc_max, experiments_auprc_max['AGO2_CLASH_Hejret2023']], axis=1)

    sota_df['metric'] = sota_df['ACC_max'].fillna(sota_df['AUPRC_max'])
    sota_df['method'] = 'Human SOTA'
    sota_df_agg = sota_df.pivot(columns="dataset", values="metric", index="method")

    max_and_sota_df = pd.concat([max_metrics_df, sota_df_agg], axis=0, join='outer', ignore_index=False)
    max_and_sota_df.to_csv("max_and_sota_df.csv", index=True)

    # MEAN metrics
    experiments_acc_mean = experiments_df.pivot(columns="dataset", values="ACC_mean", index="method")
    experiments_auprc_mean = experiments_df.pivot(columns="dataset", values="AUPRC_mean", index="method")
    experiments_acc_mean = experiments_acc_mean.drop(columns=["AGO2_CLASH_Hejret2023"])
    mean_metrics_df = pd.concat([experiments_acc_mean, experiments_auprc_mean['AGO2_CLASH_Hejret2023']], axis=1)
    mean_metrics_df.to_csv("mean_metrics_df.csv", index=True)
    
    # SUCCESS RATES
    experiments_succ_rates = experiments_df.pivot(columns="dataset", values="successful_run_mean", index="method")
    experiments_succ_rates.to_csv("experiments_succ_rates.csv", index=True)

    
    
    
    

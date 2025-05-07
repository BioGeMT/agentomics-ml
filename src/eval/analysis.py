import pandas as pd
import wandb
from dotenv import load_dotenv

def get_wandb_runs_df(tags, entity='ceitec-ai', project='Agentomics-ML'):
    # Tags are joined with OR (one hit is enough to be included)

    load_dotenv()
    api = wandb.Api(timeout=60)
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": tags}}
    )
    runs_data = []
    for run in runs:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "created_at": run.created_at,
            "state": run.state,
        }
        for key, value in run.config.items():
            run_data[f"{key}"] = value
        
        for key, value in run.summary.items():
            if(key in ["AUPRC", "AUROC", "ACC", "input_tokens", "output_tokens", "total_tokens", "inference_stage", "conda_creation_failed"]):
                run_data[f"{key}"] = value
        
        runs_data.append(run_data)
    df = pd.DataFrame(runs_data)
    df['successful_run'] = df['inference_stage'].apply(lambda x: 1 if x == 2 else 0)
    print(f"Found {len(df)} runs matching the specified tags {tags}")
    
    return df

def analyse_tag(tags, name):
    df = get_wandb_runs_df(tags)
    df['conda_creation_failed'] = df['conda_creation_failed'].apply(lambda x: 1 if x == 1 else 0)
    # Only successful runs
    successes_df = df[df['successful_run'] == 1].groupby(['dataset', 'model']).agg({
        'AUPRC': ['mean', 'max'],
        'AUROC': ['mean', 'max'],
        'ACC': ['mean', 'max'],
        'successful_run': ['mean'],
        'run_id': ['nunique'],
        # 'metric_input_tokens': ['mean', 'max'],
        # 'metric_output_tokens': ['mean', 'max'],
        # 'metric_total_tokens': ['mean', 'max'],
    }).reset_index()
    # All runs
    all_df = df.groupby(['dataset', 'model']).agg({
        'successful_run': ['mean'],
        'conda_creation_failed': ['mean'],
        'run_id': ['nunique'],
    }).reset_index()

    complete_df = successes_df.merge(all_df, on=['dataset', 'model'], suffixes=('_nonfails', ''), how='outer')
    complete_df['method'] = f'{name}:' + complete_df['model']
    complete_df.columns = complete_df.columns.to_flat_index().map('_'.join).str.strip('_')
    return complete_df

def transform_leaderboard_to_long_format():
    df = pd.read_csv('./GB_leaderboard_top1_accs.csv')
    model_names = df.columns[1:]  # All column names except the first one

    # Look for metadata rows by their specific text identifiers instead of fixed indices
    article_type_row = df[df['Dataset / Model'].str.contains('Article type', na=False)].iloc[0]
    github_row = df[df['Dataset / Model'] == 'Github'].iloc[0]
    model_availability_row = df[df['Dataset / Model'] == 'Model availability'].iloc[0]
    peer_reviewed_row = df[df['Dataset / Model'] == 'Peer-reviewed'].iloc[0]
    
    model_metadata = {}
    for model in model_names:
        model_metadata[model] = {
            'article_type_date': article_type_row[model],
            'github': github_row[model],
            'model_availability': model_availability_row[model],
            'peer_reviewed': peer_reviewed_row[model],
        }
    
    metadata_rows = ['Article type', 'Github', 'Model availability', 'Peer-reviewed']
    dataset_rows = df[~df['Dataset / Model'].str.contains('|'.join(metadata_rows), na=False)]
    transformed_data = []
    for _, row in dataset_rows.iterrows():
        dataset_name = row['Dataset / Model']
        for model in model_names:
            # Skip if accuracy is NaN
            if pd.isna(row[model]):
                continue
                
            transformed_data.append({
                'dataset': dataset_name,
                'method': model,
                'accuracy': row[model],
                'article_type_date': model_metadata[model]['article_type_date'],
                'github': model_metadata[model]['github'],
                'model_availability': model_metadata[model]['model_availability'],
                'peer_reviewed': model_metadata[model]['peer_reviewed']
            })
    
    transformed_df = pd.DataFrame(transformed_data)
    # transformed_df.to_csv('./leaderboard_long_format.csv', index=False)
    return transformed_df

def process_leaderboard(df):
    df = df[(df['peer_reviewed'] == 'yes') & (df['github'] == 'yes')]
    df = df.loc[df.groupby('dataset')['accuracy'].idxmax()]
    old_name_to_new_name = {
        "Coding vs Intergenomic": "demo_coding_vs_intergenomic_seqs",
        "Human Enhancers Cohn":"human_enhancers_cohn",
        "Human Enhancers Ensembl":"human_enhancers_ensembl",
        "Human NonTATA Promoters":"human_nontata_promoters",
        "Human OCR Ensembl":"human_ocr_ensembl",
        "Human Regulatory":"human_ensembl_regulatory",
        "Human vs Worm":"demo_human_or_worm",
        "Mouse Enhancers":"dummy_mouse_enhancers_ensembl",
        "Drosophila Enhancers Stark":"drosophila_enhancers_stark",
    }
    df['dataset'] = df['dataset'].replace(old_name_to_new_name)
    df = df[['dataset', 'method', 'accuracy']]
    df.rename(columns={'accuracy': 'ACC_max'}, inplace=True)
    return df

def add_best_tags(df):
    for col in df.columns:
        df[col] = df[col].astype(float)
        best_method = df[col].idxmax()
        non_human_has_best = df[col][~df.index.str.contains('human_SOTA')].notna().any()
        if non_human_has_best:
            best_nonhuman_method = df[col][~df.index.str.contains('human_SOTA')].idxmax()
        df[col] = df[col].astype(str)
        df.loc[best_method, col] = f"{df.loc[best_method, col]} (BEST)"
        if non_human_has_best:
            df.loc[best_nonhuman_method, col] = f"{df.loc[best_nonhuman_method, col]} (BEST NON-HUMAN)"
    return df


if __name__ == "__main__":
    # Experiment data
    sub_dfs = []
    for tag_group, name in [
        (['andrea_run_one_shot_v3'], 'single_pass'), 
        # (['testing', 'any']),
    ]:
        sub_df = analyse_tag(tags=tag_group, name=name)
        sub_dfs.append(sub_df)    
    experiment_dfs = pd.concat(sub_dfs, axis=0, join='outer')
    experiment_dfs.to_csv('./experiment_dfs.csv', index=False)

    # Genomic benchmarks and ACC_max dataframe
    gb_leaderboard_df = process_leaderboard(transform_leaderboard_to_long_format())
    gb_leaderboard_df['method'] = 'human_SOTA'
    gb_df = pd.concat([experiment_dfs, gb_leaderboard_df], axis=0, join='inner')
    gb_datasets = [
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_nontata_promoters",
        "human_ocr_ensembl",
        # "human_ensembl_regulatory",
        "drosophila_enhancers_stark"
    ]
    gb_df = gb_df[gb_df['dataset'].isin(gb_datasets)]
    gb_df_acc_max = gb_df.pivot(columns="dataset", values="ACC_max", index="method")

    # Mean metrics dataframes
    experiments_acc_mean = experiment_dfs.pivot(columns="dataset", values="ACC_mean", index="method")
    experiments_auprc_mean = experiment_dfs.pivot(columns="dataset", values="AUPRC_mean", index="method")
    experiments_succ_rates = experiment_dfs.pivot(columns="dataset", values="successful_run_mean", index="method")
    
    # average over all datasets (columns)
    experiments_succ_rates['ALL'] = experiments_succ_rates.mean(axis=1)

    experiments_acc_mean.to_csv('./experiments_acc_mean.csv', index=True)
    experiments_auprc_mean.to_csv('./experiments_auprc_mean.csv', index=True)
    experiments_succ_rates.to_csv('./experiments_succ_rates.csv', index=True)

    # AGO hejret and AUPRC_max dataframe
    ago_leaderboard_df = pd.DataFrame({
        'method': ['human_SOTA'], # mirbind retrained
        'dataset': ['AGO2_CLASH_Hejret2023'],
        'AUPRC_max': [0.86],
    })
    ago_df = pd.concat([experiment_dfs, ago_leaderboard_df], axis=0, join='inner')
    ago_df = ago_df[ago_df['dataset'].isin(['AGO2_CLASH_Hejret2023'])]
    ago_df_auprc_max = ago_df.pivot(columns="dataset", values="AUPRC_max", index="method")

    # Combined 
    max_df = pd.concat([gb_df_acc_max, ago_df_auprc_max], axis=1, join='outer')

    # Adding BEST tags to cells for visual clarity
    max_df = add_best_tags(max_df)
    max_df.to_csv('./leaderboard_max_metric.csv', index=True)
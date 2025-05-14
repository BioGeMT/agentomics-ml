import pandas as pd
import wandb
from dotenv import load_dotenv

def generate_noniterative_runs(tags, path, method, drop_na, entity='ceitec-ai', project='Agentomics-ML'):
    load_dotenv()
    api = wandb.Api(timeout=120)
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": tags}},
    )
    runs_data = []
    for run in runs:
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "created_at": run.created_at,
            "state": run.state, #necessary?
        }
        for key, value in run.config.items():
            run_data[f"{key}"] = value
        for key, value in run.summary.items():
            if(key in ["AUPRC", "AUROC", "ACC", "inference_stage"]):
                run_data[f"{key}"] = value
        runs_data.append(run_data)

    df = pd.DataFrame(runs_data)
    if(drop_na): # drops killed
        df = df.dropna().reset_index(drop=True)

    df['successful_run'] = df['inference_stage'].apply(lambda x: True if x == 2 else False)
    print(f"Found {len(df)} runs matching the specified tags {tags}")
    cols_to_drop = [ 
        'run_id', 
        'tags', 
        'agent_id',
        'state',
        'inference_stage',
    ]
    df = df.drop(columns=cols_to_drop)
    df['method'] = method + ":" + df['model']
    df.to_csv(path, index=False)
    return df

def generate_iterative_runs(tags, path, method, entity='ceitec-ai', project='Agentomics-ML'):
    load_dotenv()
    api = wandb.Api(timeout=120)
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
            if(key in ["AUPRC", "AUROC", "ACC", "inference_stage"]):
                run_data[f"{key}"] = value
            
        metrics = ['ACC', 'AUPRC', 'AUROC']
        val_histories = [f'validation/{metric}' for metric in metrics]
        train_histories = [f'train/{metric}' for metric in metrics]
        stealth_test_histories = [f'stealth_test/{metric}' for metric in metrics]
        histories = val_histories + train_histories + stealth_test_histories
        history_data = run.history(keys=histories, pandas=True)
        # if run is not finished
        if(not history_data.empty):
            for metric in metrics:
                best_val_row_index = history_data[f'validation/{metric}'].idxmax()
                max_val_row = history_data.loc[best_val_row_index]
                test_val = max_val_row[f'stealth_test/{metric}']
                run_data[f'{metric}'] = test_val
                run_data[f'best_{metric}_iteration'] = best_val_row_index

                zeroth_iteration_row = history_data.loc[0]
                test_val_zeroth = zeroth_iteration_row[f'stealth_test/{metric}']
                run_data[f'{metric}_zeroth'] = test_val_zeroth

                #TODO compute max of 0th iterations (5 runs) -> compare to average best metric over 5 iterations to estimate how much feedback helps
                run_data[f'{metric}_gain_on_zeroth'] = test_val - test_val_zeroth

        runs_data.append(run_data)

    df = pd.DataFrame(runs_data)
    df['successful_run'] = df['ACC'].apply(lambda x: True if x >= 0 else False)
    df = df.rename(columns={'best_metric': 'metric_to_optimize'})
    cols_to_drop = [ 
        'run_id', 
        'tags', 
        'agent_id',
        # 'state',
        'inference_stage',
        'use_proxy',
        'credit_budget',
        'max_run_retries',
    ]
    df = df.drop(columns=cols_to_drop)
    df['method'] = method + ":" + df['model']
    df = df[df['model'] == 'GPT4_1']
    df.to_csv(path, index=False)
    print(f"Found {len(df)} runs matching the specified tags {tags}")
    return df

def transform_gb_leaderboard_to_long_format(path):
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
    transformed_df.to_csv(path, index=False)
    return transformed_df

def process_leaderboard(leaderboard_path, save_path):
    df = pd.read_csv(leaderboard_path)
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
    used_datasets = [
        "human_enhancers_cohn",
        "human_enhancers_ensembl",
        "human_nontata_promoters",
        "human_ocr_ensembl",
        "drosophila_enhancers_stark" #Missing
    ]
    df = df[df['dataset'].isin(used_datasets)]

    df = df[['dataset', 'method', 'accuracy']]
    df.rename(columns={'accuracy': 'ACC_max'}, inplace=True)
    df['ACC_max'] = df['ACC_max'] / 100

    row = {
        'dataset': 'AGO2_CLASH_Hejret2023',
        'method': 'miRBench',
        'AUPRC_max': 0.86,
    }
    # add row
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(save_path, index=False)
    return df

if __name__ == "__main__":
    _ = generate_noniterative_runs(
        tags=['andrea_run_one_shot_v3', 'andrea_run_one_shot_v4'], 
        path="zeroshot_runs.csv",
        method="zero_shot",
        drop_na=True,
    )
    _ = generate_noniterative_runs(
        tags=['andrea_DI_v2'], 
        path="DI_runs.csv",
        method="DI",
        drop_na=True,
    )
    _ = generate_iterative_runs(
        tags=['agentomics_v10', 'agentomics_v11'], 
        path="Agentomics_runs.csv",
        method="Agentomics",
    )
    transform_gb_leaderboard_to_long_format('./gb_leaderboard_long_format.csv')
    process_leaderboard('./gb_leaderboard_long_format.csv', './SOTA_leaderboard.csv')

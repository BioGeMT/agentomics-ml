import pandas as pd
import wandb
from dotenv import load_dotenv
import numpy as np
from weave.trace_server.trace_server_interface import CallsFilter
import pandas as pd
import wandb
import weave
from dotenv import load_dotenv

def generate_iterative_runs(tags, path, method, entity='ceitec-ai', project='Agentomics-ML'):
    load_dotenv()
    api = wandb.Api(timeout=120)
    weave_client = weave.init(f"{entity}/{project}")
    runs = api.runs(
        f"{entity}/{project}",
        # filters={"tags": {"$in": tags}}
    )
    runs_data = []
    for run in runs:
        if('_runtime' not in run.summary.keys()):
            continue
        if run.id != 're4ynxfb': #TODO remove
            continue
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "created_at": run.created_at,
            "state": run.state,
            'wandb_runtime': run.summary['_runtime'],
            "method":method,
            **get_run_tokens_info(entity=entity, project=project, run_id=run.id, weave_client=weave_client),
        }

        for key, value in run.config.items():
            run_data[f"{key}"] = value
        
        history_data = run.history(pandas=True)
        new_best_data = history_data['validation/new_best']
        run_data['best_iteration'] = new_best_data[new_best_data == True].last_valid_index()
        run_data['successful_run'] = run_data['best_iteration'] is not None
        
        splits = ['validation', 'train']
        # if run is finished
        all_metrics = [key.split('/')[-1] for key in history_data.keys() if key.split('/')[0] in splits]
        print('metrics', all_metrics)
        timeseries_cols = ['duration', '_timestamp', '_step', '_runtime'] + all_metrics
        if(not history_data.empty):
            sample_timeseries_field = 'validation/snapshot_reset'
            iterations_completed = len(history_data[sample_timeseries_field])
            for k,v in history_data.items():
                if (k.split('/')[0] not in splits and k not in timeseries_cols): #split data and timeseries will get processed separately
                    run_data[k]=v
                else:
                    assert len(v) == iterations_completed, f'{k} is not of length {iterations_completed}'
                    for i, v in enumerate(history_data[k]):
                        if k in all_metrics:
                            run_data[f'{i}:test/{k}'] = v
                        else:
                            run_data[f'{i}:{k}'] = v

        runs_data.append(run_data)

    df = pd.DataFrame(runs_data)
    df = df.rename(columns={'val_metric': 'metric_to_optimize'})
    cols_to_drop = [ 
        'run_id', 
        'tags', 
        'agent_id',
        'inference_stage',
        'use_proxy',
        'credit_budget',
    ]
    df = df.drop(columns=cols_to_drop)
    df['models'] = df['model_name'] +' / '+ df['feedback_model_name']

    df.to_csv(path, index=False)
    print(f"Found {len(df)} runs matching the specified tags {tags}")
    return df

def get_run_tokens_info(entity, project, run_id, weave_client):
    run_info = {
        'prompt_tokens': None,
        'completion_tokens': None,
        'requests': None,
        'total_tokens': None,
        'reasoning_tokens': None
    }
    calls = weave_client.get_calls(filter=CallsFilter(wb_run_ids=[f"{entity}/{project}/{run_id}"], trace_roots_only=True))
    assert len(calls) == 1, len(calls)
    call = calls[0]
    # print(call.summary)
    usage = call.summary.get('usage')
    
    # sum across all models
    if not usage:
        return run_info
    for model, info in usage.items():
        simple_fields = ['prompt_tokens', 'completion_tokens', 'requests', 'total_tokens']
        for simple_field in simple_fields:
            if run_info[simple_field] is None:
                run_info[simple_field] = 0
            run_info[simple_field] += info.get(simple_field, 0)
            
        ctd = info.get('completion_tokens_details')
        reasoning_tokens = ctd.get('reasoning_tokens') if ctd is not None else 0
        if run_info['reasoning_tokens'] is None:
            run_info['reasoning_tokens'] = 0

        run_info['reasoning_tokens'] += reasoning_tokens
        
    return run_info

def generate_aide_runs(tags, path, method, drop_na, entity='ceitec-ai', project='Agentomics-ML'):
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
            "model": "gpt-4.1-2025-04-14",
            'duration': -1,

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
        'aide_eval_strategy_arg_value_intended',
    ]
    df = df.drop(columns=cols_to_drop)
    df['method'] = method + ":" + df['model']
    df.to_csv(path, index=False)
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
    
    transformed_data.append({
        'dataset': 'Drosophila Enhancers Stark',
        'method': 'Genomic benchmarks',
        'accuracy': 58.6,
        'article_type_date': '',
        'github': 'yes',
        'model_availability': '',
        'peer_reviewed': 'yes',
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
        "drosophila_enhancers_stark"
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
    # _ = generate_noniterative_runs(
    #     tags=['andrea_run_one_shot_v3', 'andrea_run_one_shot_v4', 'oneshot_recover_v1'], 
    #     path="zeroshot_runs.csv",
    #     method="zero_shot",
    #     drop_na=True,
    # )
    # _ = generate_noniterative_runs(
    #     tags=['andrea_DI_v2', 'DI_recover_v1'], 
    #     path="DI_runs.csv",
    #     method="DI",
    #     drop_na=True,
    # )
    _ = generate_iterative_runs(
        tags=['agentomics_v10', 'agentomics_v11', 'agentomics_einfra_v1'], 
        path="Agentomics_runs.csv",
        method="Agentomics",
    )
    _ = generate_aide_runs(
        tags=['vlasta_aide_v3'], 
        path="AIDE_runs.csv",
        method="AIDE",
        drop_na=True,
    )
    transform_gb_leaderboard_to_long_format('./gb_leaderboard_long_format.csv')
    process_leaderboard('./gb_leaderboard_long_format.csv', './SOTA_leaderboard.csv')

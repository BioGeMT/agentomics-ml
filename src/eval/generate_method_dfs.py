import pandas as pd
import wandb
from dotenv import load_dotenv
import numpy as np
from weave.trace_server.trace_server_interface import CallsFilter
import pandas as pd
import wandb
import weave
from dotenv import load_dotenv
import math

def dataset_type_to_domain(row):
    dt_map = {
        "Protein":"Protein Engineering",
        "Small Molecule":"Drug Discovery",
        "DNA/RNA":"Regulatory Genomics",
    }
    return dt_map[row['dataset_type']]

def get_clean_data_factory(history_data, iterations_completed):
    def get_clean_data(col_name, dict_mode=False, alt_step_axis=None):
        if(alt_step_axis is not None):
            sub_result = history_data[col_name]
            sub_result = sub_result[pd.notna(sub_result)] #removing nans
            indicies = sub_result.index.tolist()
            iteration_index = history_data[alt_step_axis].loc[indicies]
            if dict_mode:
                return {iteration:value for iteration,value in zip(iteration_index.values, sub_result.values)}
            return sub_result.values
        
        result = history_data[col_name][:iterations_completed]

        result = result[pd.notna(result)] #removing nans
        for v in history_data[col_name][iterations_completed:]:
            assert pd.isna(v), f"Expected NaN values after iterations_completed in {col_name}"
        if dict_mode:
            return {iteration:value for iteration, value in zip(result.index, result.values)}
        return result
    return get_clean_data

def get_singular_value(series):
    valid_values = [v for v in series if not pd.isna(v)]
    assert len(set(valid_values)) == 1, f"Expected 1 non-NaN value, found {len(valid_values)}"
    return valid_values[0]

def is_numeric(s):
    if s=='NaN':
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

#TODO redo this into taking dictionary with actual iteration number
def compute_best_val_metric_so_far(ordered_val_metrics_array, metric_name):
    if metric_name in ['RMSE','MAE','LOG_LOSS']:
        higher_is_better = False
    elif metric_name in ['AUPRC','AUROC','ACC','PEARSON','SPEARMAN']:
        higher_is_better = True
    else:
        raise Exception('unknown metric', metric_name)
    best_so_far = []
    current_best = None
    for val in ordered_val_metrics_array:
        if val is not None:
            val = float(val)
        if current_best is None:
            current_best = val
        else:
            if pd.isna(val):
                pass
            else:
                if higher_is_better:
                    if val > current_best:
                        current_best = val
                else:
                    if val < current_best:
                        current_best = val
        best_so_far.append(float(current_best))
    return best_so_far

def get_dataset_type(dataset):
    gb_names = ['AGO2_CLASH_Hejret2023', 'drosophila_enhancers_stark', 'human_enhancers_cohn', 'human_enhancers_ensembl', 'human_ocr_ensembl']
    drug_names = ['adme-fang-SOLU-1', 'pkis2-egfr-wt-c-1', 'caco2-wang', 'lipophilicity-astrazeneca', 'adme-fang-HPPB-1', 'adme-fang-HCLint-1', 'bbb-martins', 'cyp2d6-substrate-carbonmangels', 'herg']
    prot_names = ['SPA_STAAU', 'CBX4_HUMAN_Tsuboyama_2023_2K28', 'CSN4_MOUSE_Tsuboyama_2023_1UFM_indels', 'PSAE_PICP2_Tsuboyama_2023_1PSE', 'Q8EG35_SHEON_Campbell_2022_indels', 'SBI_STAAM_Tsuboyama_2023_2JVG', 'SPIKE_SARS2_Starr_2020_binding']
    if any(name in dataset for name in gb_names):
        return 'DNA/RNA'
    elif any(name in dataset for name in drug_names):
        return 'Small Molecule'
    elif any(name in dataset for name in prot_names):
        return 'Protein'
    else:
        return 'Other'

def extend_best_iters(best_iters, iters_completed):
    curr_best_iter = None
    result = []
    for i in range(iters_completed):
        if i in best_iters:
            result.append(i)
            curr_best_iter = i
        else:
            result.append(curr_best_iter)
    return result

def generate_iterative_runs(tags, path, gb_leaderboard, train_datasize_dict, test_datasize_dict, entity='ceitec-ai', project='Agentomics-ML'):
    load_dotenv()
    api = wandb.Api(timeout=120)
    weave_client = weave.init(f"{entity}/{project}")
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": tags}}
    )
    runs_data = []
    for run in runs:
        if('_runtime' not in run.summary.keys()):
            continue
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "created_at": run.created_at,
            "state": run.state,
            'wandb_runtime': run.summary['_runtime'],
            # **get_run_tokens_info(entity=entity, project=project, run_id=run.id, weave_client=weave_client),
        }

        for key, value in run.config.items():
            run_data[f"{key}"] = value
        
        history_data = run.history(pandas=True)
        iterations_completed = history_data['validation/snapshot_reset'].last_valid_index() + 1
        run_data['number_of_trainval_splits'] = (history_data['validation/snapshot_reset'].values == 1).sum()
        run_data['iterations_completed'] = iterations_completed
        get_clean_data = get_clean_data_factory(history_data, iterations_completed)

        #TODO can be used to analyse split frequency
        # val_split_changed_arr = get_clean_data('validation/snapshot_reset')
        # val_split_changed_dict = get_clean_data('validation/snapshot_reset', dict_mode=True)

        main_metric_name = run_data['val_metric']
        main_metric_validation_dict = get_clean_data(f"validation/{main_metric_name}", dict_mode=True)
        assert all([not pd.isna(v) for v in main_metric_validation_dict.values()]), main_metric_validation_dict
        if f"stealth_test/{main_metric_name}" in history_data.keys():
            main_metric_test_dict = get_clean_data(f"stealth_test/{main_metric_name}", dict_mode=True, alt_step_axis='iteration_index')
        else:
            main_metric_test_dict = None

        run_data['best_iteration_logged'] = history_data['validation/new_best'][history_data['validation/new_best'] == True].last_valid_index()
        run_data['best_iteration_at_%_of_total'] = run_data['best_iteration_logged'] / run_data['iterations_completed'] 

        run_data['best_iter_validation_mainmetric'] = main_metric_validation_dict[run_data['best_iteration_logged']]

        run_data['successful_run'] = run_data['best_iteration_logged'] is not None
        if 'api_usage/usage' in history_data.keys():
            run_data['usd_cost'] = get_singular_value(history_data['api_usage/usage'])
            run_data['cost_per_iter'] = run_data['usd_cost'] / run_data['iterations_completed']

        run_data['iterations_duration'] = get_clean_data('duration').values

        run_data['cumulative_iterations_duration'] = np.cumsum(run_data['iterations_duration']).tolist()
        assert len(run_data['cumulative_iterations_duration']) == run_data['iterations_completed']

        run_data['dataset_type'] = get_dataset_type(run_data['dataset'])
        run_data['best_iterations'] = list(history_data['validation/new_best'][history_data['validation/new_best'] == True].index.tolist())
        run_data['each_iters_best_iteration'] = extend_best_iters(run_data['best_iterations'], run_data['iterations_completed'])
        
        splits = ['validation', 'train']
        all_metrics = [key.split('/')[-1] for key in history_data.keys() if key.split('/')[0] in splits and key not in ['validation/snapshot_reset','validation/splitting_allowed', 'validation/new_best']]

        for metric_name in all_metrics:
            if metric_name not in history_data.keys():
                # print(f"WARNING {run_data['run_name']} {metric_name} not found in history data")
                continue
            run_data[f'test/{metric_name}'] = get_singular_value(history_data[metric_name])

        try:
            run_data['best_iter_test_mainmetric'] = run_data[f'test/{main_metric_name}']
        except Exception as e:
            # print('WARNING', run_data['run_name'], 'missing test metric')
            continue
        run_data['best_iter_val_test_mainmetric_diff'] = run_data['best_iter_validation_mainmetric'] - run_data['best_iter_test_mainmetric']

        if 'biomlbench/leaderboard_percentile' in history_data.keys():
            run_data['biomlbench%'] = get_singular_value(history_data['biomlbench/leaderboard_percentile'])
            run_data['human_sota'] = get_singular_value(history_data['biomlbench/gold_threshold'])
            
            run_data['biomlbench_score'] = get_singular_value(history_data['biomlbench/score'])
            try:
                assert math.isclose(run_data[f'test/{main_metric_name}'],run_data['biomlbench_score'], rel_tol=1e-3), f"{run_data[f'test/{main_metric_name}']} VS {run_data['biomlbench_score']}"
            except AssertionError as e:
                # print(f"WARNING {run_data['run_name']} test metric doesnt match!")
                pass
        else:
            run_data['human_sota'] = max(gb_leaderboard[run_data['dataset']])
        
        if main_metric_test_dict is not None:
            valid_validation_iters = [k for k,v in main_metric_validation_dict.items() if is_numeric(v)]
            valid_test_iters = [k for k,v in main_metric_test_dict.items() if is_numeric(v)]

            common_iters = list(set(valid_validation_iters) & set(valid_test_iters))
            common_iters.sort()
            if(len(common_iters) != len(main_metric_validation_dict.keys())):
                # print(f"Warning: Run {run.name} has difference in val/stealth iters {set(valid_validation_iters).difference(set(valid_test_iters))} Using only common iterations {list(common_iters)} for correlation computation.")
                pass
            val_mainmetrics = [main_metric_validation_dict[i] for i in common_iters]
            assert all([is_numeric(v) for v in val_mainmetrics]), val_mainmetrics
            test_mainmetrics = [main_metric_test_dict[i] for i in common_iters]
            run_data['common_val_test_iters_count'] = len(common_iters)
            run_data['val_stealth_mainmetric_corr'] = np.corrcoef(val_mainmetrics, test_mainmetrics)[0, 1] #pearson correlation coefficient
            run_data['stealth_test_per_iter_array_common'] = test_mainmetrics
            run_data['valid_per_iter_array_common'] = val_mainmetrics

            stealth_test_per_iter_array = []
            for i in range(run_data['iterations_completed']):
                if i in main_metric_test_dict.keys():
                    stealth_test_per_iter_array.append(main_metric_test_dict[i])
                else:
                    stealth_test_per_iter_array.append(np.nan)
            assert len(stealth_test_per_iter_array) == run_data['iterations_completed']
            run_data['stealth_test_per_iter_array'] = stealth_test_per_iter_array
            
            would_be_test = []
            for best_iter_index in run_data['each_iters_best_iteration']:
                if best_iter_index not in main_metric_test_dict:
                    # print('WARNING best iter didnt compute test!', run_data['run_name'])
                    would_be_test.append(None)
                else:
                    would_be_test.append(float(main_metric_test_dict[best_iter_index]))
            assert len(would_be_test) == run_data['iterations_completed']
            run_data['would_be_test'] = would_be_test
        
        run_data['best_val_metric_so_far'] = compute_best_val_metric_so_far(main_metric_validation_dict, main_metric_name)
        if run_data['val_metric'] in ['AUPRC','AUROC','ACC','PEARSON','SPEARMAN']:
            run_data['is_new_sota'] = float(run_data['best_iter_test_mainmetric']) > float(run_data['human_sota'])
        elif run_data['val_metric'] in ['RMSE','MAE','LOG_LOSS']:
            run_data['is_new_sota'] = float(run_data['best_iter_test_mainmetric']) < float(run_data['human_sota'])
        else:
            run_data['is_new_sota'] = None
            raise Exception('new sota none')
        
        runs_data.append(run_data)

    df = pd.DataFrame(runs_data)
    
    df = df.rename(columns={'val_metric': 'main_metric'})
    cols_to_drop = [ 
        # 'run_id', 
        # 'tags', 
        # 'agent_id',
        # 'inference_stage',
        'use_proxy',
        'credit_budget',
    ]
    df = df.drop(columns=cols_to_drop)
    # df['models'] = df['model_name'] +' / '+ df['feedback_model_name']


    df['gb%'] = df.apply(lambda x: 100*sum(x['best_iter_test_mainmetric'] > np.array(gb_leaderboard[x['dataset']]))/len(gb_leaderboard[x['dataset']]) if x['dataset'] in gb_leaderboard.keys() else None, axis=1)
    df['leaderboard%'] = df.apply(lambda x: x['gb%'] if x['dataset'] in gb_leaderboard.keys() else x['biomlbench%'],axis=1)
    df['leaderboard_based_human_sota'] = df.apply(lambda x: max(gb_leaderboard[x['dataset']]) if x['dataset'] in gb_leaderboard.keys() else x['human_sota'],axis=1)
    df['train_size'] = df.apply(lambda x: train_datasize_dict[x['dataset']],axis=1)
    df['test_size'] = df.apply(lambda x: test_datasize_dict[x['dataset']],axis=1)
    df['duration_per_iter'] = df.apply(lambda x: 8 / x['iterations_completed'],axis=1) #8 hour runs
    df['best_iteration_at_h'] = df['best_iteration_at_%_of_total'] * 8 #8 hour runs
    df['log_train_size'] = np.log(df['train_size'])
    df['log_test_size'] = np.log(df['test_size'])
    df['domain'] = df.apply(dataset_type_to_domain,axis=1) #8 hour runs

    # Filter old small molecule runs without proper FMs
    filtered=df[~((df['dataset_type'] == 'Small Molecule') & (df['tags'].apply(lambda x: 'ismb2026_v2' in x) | (df['tags'].apply(lambda x: 'ismb2026_v3' in x))))]
    filtered = filtered[filtered['run_name'] != 'miserable_avarice_look']
    filtered['is_max%_on_its_dataset'] = (
        filtered['leaderboard%']
        == filtered.groupby('dataset')['leaderboard%'].transform('max')
    )
    filtered[['run_name', 'is_max%_on_its_dataset']].to_csv('./paper_tables/best_run_names.csv')


    cols_to_keep = [
        'run_name', 'tags', 'dataset', 'task_type', 'model_name', 'main_metric', 
        'feedback_model_name',
        'iterations_completed',
        'best_iteration_logged', 'best_iter_validation_mainmetric',
        'successful_run',  'dataset_type',
        'best_iter_test_mainmetric', 'number_of_trainval_splits',
        'best_iter_val_test_mainmetric_diff',  'common_val_test_iters_count',#not used
        'human_sota', 'leaderboard_based_human_sota',
        'val_stealth_mainmetric_corr',
        'train_size','test_size', 'domain',
        'log_train_size','log_test_size',
        # 'biomlbench%','gb%',
        'is_new_sota', 'usd_cost', 'cost_per_iter',  'biomlbench_score',  'leaderboard%', 'duration_per_iter',
        'iterations_duration', 'best_iterations', 'best_val_metric_so_far', 'each_iters_best_iteration', 
        'would_be_test', 'stealth_test_per_iter_array', 'cumulative_iterations_duration', 'best_iteration_at_%_of_total', 'best_iteration_at_h',
        'stealth_test_per_iter_array_common', 'valid_per_iter_array_common', 'is_max%_on_its_dataset',
    ]
    fin_df = filtered[cols_to_keep]

    fin_df.to_csv(path, index=False)
    # print(f"Found {len(df)} runs matching the specified tags {tags}")
    return fin_df

def get_run_tokens_info(entity, project, run_id, weave_client):
    run_info = {
        'prompt_tokens': None,
        'completion_tokens': None,
        'requests': None,
        'total_tokens': None,
        'reasoning_tokens': None
    }
    calls = weave_client.get_calls(filter=CallsFilter(wb_run_ids=[f"{entity}/{project}/{run_id}"], trace_roots_only=True))
    if(len(calls) != 1):
        return {}

    assert len(calls) == 1, len(calls)
    call = calls[0]
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


def generate_zeroshot_runs(tags, path, entity='ceitec-ai', project='Agentomics-ML'):
    load_dotenv()
    metrics_of_interest = ['ACC', 'AUPRC']
    api = wandb.Api(timeout=120)
    runs = api.runs(
        f"{entity}/{project}",
        filters={"tags": {"$in": tags}}
    )
    runs_data = []
    for run in runs:
        if('_runtime' not in run.summary.keys()):
            continue
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "tags": run.tags,
            "created_at": run.created_at,
            "state": run.state,
            'wandb_runtime': run.summary['_runtime'],
            # 'val_metric': run.val_metric,
            # **get_run_tokens_info(entity=entity, project=project, run_id=run.id, weave_client=weave_client),
        }
        history_data = run.history(pandas=True)

        for key, value in run.config.items():
            run_data[f"{key}"] = value

        # main_metric_name = run_data['val_metric']
        for k,v in history_data.items():
            if k in metrics_of_interest:
                run_data[k] = float(get_singular_value(v))
                if run_data[k] == -1:
                    run_data[k] = None
                    run_data['successful_run'] = False
                else:
                    run_data['successful_run'] = True

        # run_data['best_mainmetric'] = get_singular_value(history_data[main_metric_name])
        runs_data.append(run_data)
    df = pd.DataFrame(runs_data)

    df.to_csv(path, index=False)
    return df


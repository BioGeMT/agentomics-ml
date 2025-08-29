#!/usr/bin/env python3
import json
import shutil
import pandas as pd
from pathlib import Path

def auto_detect_target_col(train_df):
    """Auto-detect target column"""
    possible_target_cols = ['class', 'target', 'label', 'y']
    for col in possible_target_cols:
        if col in train_df.columns:
            print(f'INFO: Auto-detected target column: {col}')
            return col
        
    print(f'INFO: Using last column as target: {train_df.columns[-1]}')
    return train_df.columns[-1]
        
def auto_detect_task_type(train_df, target_col) :
    """Auto-detect task type based on target column values"""
    target_values = train_df[target_col].dropna()
    unique_values = target_values.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_values)
    
    if is_numeric and unique_values > 10:
        print(f'INFO: Auto-detected regression task (numeric target with {unique_values} unique values)')
        return 'regression'
    
    print(f'INFO: Auto-detected classification task ({unique_values} unique values)')
    return 'classification'

def get_label_to_number_map(train_df, test_df, target_col, positive_class=None, negative_class=None):
    unique_labels = train_df[target_col].dropna().unique()

    # Check test set for additional labels
    if test_df is not None:
        test_unique_labels = test_df[target_col].dropna().unique()
        if(set(unique_labels) != set(test_unique_labels)):
            print("WARNING: Mismatch in unique labels between train and test sets.", 
                f"Train labels: {unique_labels}, Test labels: {test_unique_labels}")
            unique_labels = set(unique_labels).union(set(test_unique_labels))

    # Generate label to number mapping
    if len(unique_labels) == 2 and positive_class and negative_class:
        # binary classification
        label_map = {negative_class: 0, positive_class: 1}
    else:
        # multiclass: alphabetical order
        label_map = {lbl: i for i, lbl in enumerate(sorted(unique_labels, key=str))}

    print(f"INFO: Label to number mapping: {label_map}. If this is wrong, please provide positive-class and negative-class parameters to the script.")

    return label_map

def prepare_dataset(dataset_dir, target_col, 
                   positive_class, negative_class, task_type, output_dir):
    """
    Preprocesses dataset files to a format digestable by the agent code
    If target_col and/or task_type is None, it will be auto-detected and printed out
    If positive_class and negative_class are None, they will be auto-detected for binary classification and printed out
    """
    train = dataset_dir / 'train.csv'
    test = dataset_dir / 'test.csv' if (dataset_dir / 'test.csv').exists() else None
    description = dataset_dir / 'dataset_description.md' if (dataset_dir / 'dataset_description.md').exists() else None
    name = dataset_dir.name

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test) if test else None
    
    if target_col is None:
        target_col = auto_detect_target_col(train_df)
    if task_type is None:
        task_type = auto_detect_task_type(train_df, target_col)
    
    if task_type == 'classification':
        label_map = get_label_to_number_map(
            train_df=train_df,
            test_df=test_df,
            target_col=target_col,
            positive_class=positive_class,
            negative_class=negative_class
        )
    
    dataframes = [('train', train_df)]
    if test_df is not None:
        dataframes.append(('test', test_df))
    
    dataset_name = name
    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate train, test, and no_label CSV files
    for split_name, df in dataframes:
        if task_type == 'classification':
            df['numeric_label'] = df[target_col].map(label_map)
        else:
            df['numeric_label'] = df[target_col]

        df.to_csv(out_dir / f'{split_name}.csv', index=False)
        df.drop([target_col, 'numeric_label'], axis=1).to_csv(
            out_dir / f'{split_name}.no_label.csv', index=False
        )
    
    # Generate dataset description file
    if(description is not None):
        description_content = description.read_text()
        (out_dir / 'dataset_description.md').write_text(description_content)
    else:
        print("INFO: No dataset description provided.")
        (out_dir / 'dataset_description.md').write_text("No dataset description available.")
    
    # Generate metadata file
    meta = {
        'task_type': task_type,
        'class_col': target_col,
        'numeric_label_col': 'numeric_label',
    }
    if task_type == 'classification':
        json_safe_label_map = {str(k): int(v) for k, v in label_map.items()}
        meta['label_to_scalar'] = json_safe_label_map

    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=4))

def setup_agent_datasets(dataset_dir, dataset_agent_dir):
    dataset_agent_dir.mkdir(parents=True, exist_ok=True)
    """
    Copies non-sensitive (non-test) dataset files to a directory accessible by agents.
    """

    # If the directories are the same, skip copying to avoid circular references
    if dataset_dir.resolve() == dataset_agent_dir.resolve():
        print(f"INFO: Dataset directory and agent directory are the same ({dataset_dir}), skipping file copying")
        return

    target_files = ['dataset_description.md', 'train.csv', 'train.no_label.csv']

    for dataset in dataset_dir.iterdir():
        if dataset.is_dir():
            agent_subfolder = dataset_agent_dir / dataset.name
            agent_subfolder.mkdir(exist_ok=True)

            for file in target_files:
                source_file = dataset / file

                target_file = dataset_agent_dir / dataset.name / file

                if source_file.exists():
                    if target_file.exists() or target_file.is_symlink():
                            target_file.unlink()

                    shutil.copy2(source_file, target_file)
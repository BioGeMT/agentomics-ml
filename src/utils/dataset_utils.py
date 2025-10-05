#!/usr/bin/env python3
import csv
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict
import subprocess

def count_csv_rows(csv_file: str) -> int:
    """
    Count rows in a CSV file (excluding header).
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            row_count = sum(1 for _ in reader)
            return max(0, row_count - 1)  # Subtract 1 for header
    except (FileNotFoundError, IOError, UnicodeDecodeError):
        return 0
    
def get_single_dataset_info(dataset_dir: str, prepared_datasets_dir: str) -> Dict:
    if not dataset_dir.is_dir():
        return None
        
    dataset_name = dataset_dir.name
    train_file = dataset_dir / "train.csv"
    test_file = dataset_dir / "test.csv"
    validation_file = dataset_dir / "validation.csv"
    
    # Count rows in raw files
    train_rows = count_csv_rows(str(train_file)) if train_file.exists() else 0
    test_rows = count_csv_rows(str(test_file)) if test_file.exists() else 0
    validation_rows = count_csv_rows(str(validation_file)) if validation_file.exists() else 0
    
    # Check if already prepared
    is_prepared = check_dataset_prepared(str(dataset_dir), prepared_datasets_dir)
    
    # Check if can be prepared
    can_prepare = train_file.exists() and train_rows > 0
    
    if not train_file.exists():
        status = "Missing train.csv"
    elif train_rows == 0:
        status = "Empty train.csv"
    elif is_prepared:
        status = "Already prepared"
    elif can_prepare:
        status = "Ready to prepare"
    else:
        status = "Cannot prepare"
        
    return {
        "name": dataset_name,
        "path": dataset_dir,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "validation_rows": validation_rows,
        "status": status,
        "can_prepare": can_prepare,
        "should_prepare": can_prepare and not is_prepared,
        "is_prepared": is_prepared
    }

def get_single_prepared_dataset_info(prepared_dataset_dir: str) -> Dict:
    if not prepared_dataset_dir.is_dir():
        return None
        
    dataset_name = prepared_dataset_dir.name
    train_file = prepared_dataset_dir / "train.csv"
    test_file = prepared_dataset_dir / "test.csv"
    validation_file = prepared_dataset_dir / "validation.csv"

    # Count rows in raw files
    train_rows = count_csv_rows(str(train_file)) if train_file.exists() else 0
    test_rows = count_csv_rows(str(test_file)) if test_file.exists() else 0
    validation_rows = count_csv_rows(str(validation_file)) if validation_file.exists() else 0
    
    if not train_file.exists():
        status = "Missing train.csv"
    elif train_rows == 0:
        status = "Empty train.csv"
    else:
        status = "Prepared"
        
    return {
        "name": dataset_name,
        "path": prepared_dataset_dir,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "validation_rows": validation_rows,
        "status": status
    }

def get_all_datasets_info(datasets_dir: str, prepared_datasets_dir: str) -> List[Dict]:
    """
    Collect information about all datasets for preparation.
    
    Args:
        datasets_dir: Path to raw datasets directory
        prepared_datasets_dir: Path to prepared datasets directory
        
    Returns:
        List of dataset information dictionaries
    """
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        return []
    
    datasets_info = []
    
    for dataset_dir in datasets_path.iterdir():
        dataset_info = get_single_dataset_info(dataset_dir, prepared_datasets_dir)
        if(dataset_info):
            datasets_info.append(dataset_info)
        
    # Sort by name for consistent ordering
    datasets_info.sort(key=lambda x: x["name"])
    return datasets_info

def get_all_prepared_datasets_info(prepared_datasets_dir: str) -> List[Dict]:
    """
    Collect information about all prepared datasets.
    
    Args:
        prepared_datasets_dir: Path to prepared datasets directory
        
    Returns:
        List of dataset information dictionaries
    """
    prepared_datasets_path = Path(prepared_datasets_dir)
    
    if not prepared_datasets_path.exists():
        return []
    
    prepared_datasets_info = []
    
    for prepared_dataset_dir in prepared_datasets_path.iterdir():
        dataset_info = get_single_prepared_dataset_info(prepared_dataset_dir)
        if(dataset_info):
            prepared_datasets_info.append(dataset_info)
        
    # Sort by name for consistent ordering
    prepared_datasets_info.sort(key=lambda x: x["name"])
    return prepared_datasets_info

def check_dataset_prepared(dataset_dir: str, prepared_datasets_dir: str) -> bool:
    """Check if a dataset is already prepared."""
    dataset_name = Path(dataset_dir).name
    prepared_path = Path(prepared_datasets_dir) / dataset_name
    metadata_file = prepared_path / "metadata.json"
    train_file = prepared_path / "train.csv"
    return metadata_file.exists() and train_file.exists()

def auto_detect_target_col(train_df):
    """Auto-detect target column"""
    possible_target_cols = ['class', 'target', 'label', 'y']
    for col in possible_target_cols:
        if col in train_df.columns:
            print(f'INFO: Auto-detected target column: {col}')
            return col
        
    print(f'INFO: Using last column as target: {train_df.columns[-1]}')
    return train_df.columns[-1]

def get_task_type_from_prepared_dataset(prepared_dataset_dir: str) -> str:
    metadata_path = prepared_dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    return metadata.get("task_type")

def get_classes_integers(config):
    """Get classes integers from the prepared dataset metadata."""
    if config.task_type != "classification":
        return None

    metadata_path = config.prepared_dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Sort by numeric value to get consistent ordering
    return sorted(metadata["label_to_scalar"].values())
        
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

def smart_sort_labels(labels):
    """
    Sort labels with semantic sense:
    - Pure numeric labels: "1", "2", "10" → 1, 2, 10
    - Labels with numeric prefixes: "Rank 1", "Rank 2", "Rank 10" → Rank 1, Rank 2, Rank 10
    - Mixed labels: "A", "B", "1", "2", "10" → 1, 2, 10, A, B
    """
    def sort_key(label):
        # Try to extract numeric part for sorting
        import re
        
        # Check if the entire label is numeric
        if str(label).strip().replace('.', '').replace('-', '').isdigit():
            return (0, float(label))  # Numeric labels first, sorted by value
        
        # Check if label has numeric prefix (e.g., "Rank 1", "Class 10")
        numeric_match = re.match(r'^(.+?)\s*(\d+(?:\.\d+)?)\s*$', str(label).strip())
        if numeric_match:
            prefix, number = numeric_match.groups()
            return (1, prefix, float(number))  # Labels with numeric suffix, sorted by prefix then number
        
        # Check if label has numeric suffix (e.g., "1st", "2nd", "10th")
        suffix_match = re.match(r'^(\d+(?:\.\d+)?)\s*(.+?)$', str(label).strip())
        if suffix_match:
            number, suffix = suffix_match.groups()
            return (1, float(number), suffix)  # Labels with numeric prefix, sorted by number then suffix
        
        # Fallback to alphabetical for non-numeric labels
        return (2, str(label))
    
    return sorted(labels, key=sort_key)

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
        # multiclass: smart semantic sorting
        sorted_labels = smart_sort_labels(unique_labels)
        label_map = {lbl: i for i, lbl in enumerate(sorted_labels)}

    print(f"INFO: Label to number mapping: {label_map}. If this is wrong, please provide positive-class and negative-class parameters to the script.")

    return label_map

def prepare_dataset(dataset_dir, target_col, 
                   positive_class, negative_class, task_type, output_dir):
    """
    Preprocesses dataset files to a format digestable by the agent code
    If target_col and/or task_type is None, it will be auto-detected and printed out
    If positive_class and negative_class are None, they will be auto-detected for binary classification and printed out
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    train = dataset_dir / 'train.csv'
    test = dataset_dir / 'test.csv' if (dataset_dir / 'test.csv').exists() else None
    validation = dataset_dir / 'validation.csv' if (dataset_dir / 'validation.csv').exists() else None
    description = dataset_dir / 'dataset_description.md' if (dataset_dir / 'dataset_description.md').exists() else None
    name = dataset_dir.name

    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test) if test else None
    validation_df = pd.read_csv(validation) if validation else None
    
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
    if validation_df is not None:
        dataframes.append(('validation', validation_df))
    
    dataset_name = name
    out_dir = Path(output_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate train, test, and no_label CSV files
    for split_name, df in dataframes:
        try:
            if task_type == 'classification':
                df['numeric_label'] = df[target_col].map(label_map)
            else:
                df['numeric_label'] = df[target_col]
        except KeyError as e:
            raise KeyError(f"Target column '{target_col}' not found in {split_name} dataset. Available columns: {df.columns}") from e

        df.drop(columns=[target_col]).to_csv(out_dir / f'{split_name}.csv', index=False)
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

    non_sensitive_files = ['dataset_description.md', 'train.csv', 'train.no_label.csv']
    for file in out_dir.iterdir():
        if file.name not in non_sensitive_files:
            subprocess.run(["chmod", "o-rwx", file], check=True)

def setup_nonsensitive_dataset_files_for_agent(prepared_datasets_dir: Path, agent_datasets_dir: Path, dataset_name: str):
    """
    Copies non-sensitive (non-test) dataset files to a shared directory accessible by agents.
    """
    source_dataset_dir = prepared_datasets_dir / dataset_name
    target_dataset_dir = agent_datasets_dir / dataset_name
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    assert target_dataset_dir.is_dir()

    target_files = ['dataset_description.md', 'train.csv', 'train.no_label.csv', 'validation.csv', 'validation.no_label.csv']
    for file in target_files:
        source_file = source_dataset_dir / file
        target_file = target_dataset_dir / file

        if source_file.exists():
            if target_file.exists() or target_file.is_symlink():
                    target_file.unlink()

            #TODO why was this changed from a symlink to a copy?
            #TODO bug was this: if agent changed the file in it's own workspace folder, it was changing the OG prepared files
            #TODO make it read-only simlink 
            #TODO check raw data -> prepared data is not a symlink but a hard copy!
            shutil.copy2(source_file, target_file)
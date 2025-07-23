#!/usr/bin/env python3
import json
import pandas as pd

def prepare_dataset(train, test, target_col, description, name, 
                   positive_class, negative_class, task_type, output_dir):
    """
    Preprocesses dataset files to a format digestable by the agent code
    """
   
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test) if test else None
    
    label_col = target_col
    
    all_labels = train_df[label_col].dropna()
    if test_df is not None:
        all_labels = pd.concat([all_labels, test_df[label_col].dropna()])
    else:
        print("INFO: No test set found.")
    
    if task_type == 'classification':
        labels = list(all_labels.unique())
    
        if len(labels) == 2 and positive_class and negative_class:
            # binary classification 
            label_map = {negative_class: 0, positive_class: 1}
            print(f"INFO: Label to number mapping: {label_map}. If this is wrong, please provide positive-class and negative-class parameters to the prepare_dataset script.")
        else:
            # multiclass: alphabetical order
            label_map = {lbl: i for i, lbl in enumerate(sorted(labels, key=str))}
            print(f"INFO: Label to number mapping: {label_map}")
    
    dataframes = [('train', train_df)]
    if test_df is not None:
        dataframes.append(('test', test_df))
    
    dataset_name = name
    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in dataframes:
        if task_type == 'classification':
            df['numeric_label'] = df[label_col].map(label_map)
        else:
            df['numeric_label'] = df[label_col]

        df.to_csv(out_dir / f'{split_name}.csv', index=False)
        df.drop([label_col, 'numeric_label'], axis=1).to_csv(
            out_dir / f'{split_name}.no_label.csv', index=False
        )
    
    if(description is not None):
        description_content = description.read_text()
        (out_dir / 'dataset_description.md').write_text(description_content)
    else:
        print("INFO: No dataset description provided.")
        (out_dir / 'dataset_description.md').write_text("No dataset description available.")
    
    meta = {
        'task_type': task_type,
        'class_col': label_col,
        'numeric_label_col': 'numeric_label'
    }

    if task_type == 'classification':
        #  numpy types to native python types for JSON 
        json_safe_label_map = {str(k): int(v) for k, v in label_map.items()}
        meta['label_to_scalar'] = json_safe_label_map
    
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=4))

def setup_agent_datasets(dataset_dir, dataset_agent_dir):
    dataset_agent_dir.mkdir(parents=True, exist_ok=True)
    """
    Links non-sentitive (non-test) dataset files to a directory accessible by agents.
    """

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

                    target_file.symlink_to(source_file)


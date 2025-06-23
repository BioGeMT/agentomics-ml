#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Import dataset")
    parser.add_argument('--train', required=True, type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--class-col', required=True, dest='class_col')
    parser.add_argument('--description', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--positive-class')
    parser.add_argument('--negative-class')
    parser.add_argument('--output-dir', type=Path, default=Path('datasets'))
    return parser.parse_args()

def prepare_dataset(train, test, class_col, description, name, 
                   positive_class, negative_class, output_dir):
   
    train_df = pd.read_csv(train)
    test_df = pd.read_csv(test) if test else None
    
    label_col = class_col
    
    all_labels = train_df[label_col].dropna()
    if test_df is not None:
        all_labels = pd.concat([all_labels, test_df[label_col].dropna()])
    labels = list(all_labels.unique())
    
    if len(labels) == 2 and positive_class and negative_class:
        # binary classification 
        label_map = {negative_class: 0, positive_class: 1}
        print(f"Binary mapping: {negative_class}=0, {positive_class}=1")
    else:
        # multiclass: alphabetical order
        label_map = {lbl: i for i, lbl in enumerate(sorted(labels, key=str))}
        print(f"Label mapping: {label_map}")
    
    dataframes = [('train', train_df)]
    if test_df is not None:
        dataframes.append(('test', test_df))
    
    dataset_name = name
    out_dir = output_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in dataframes:
        df['numeric_label'] = df[label_col].map(label_map)
        
        df.to_csv(out_dir / f'{split_name}.csv', index=False)
        
        df.drop([label_col, 'numeric_label'], axis=1).to_csv(
            out_dir / f'{split_name}.no_label.csv', index=False
        )
    
    description_content = description.read_text()
    (out_dir / 'dataset_description.md').write_text(description_content)
    
    meta = {
        'train_split': f"{output_dir}/{dataset_name}/train.csv",
        'train_split_no_labels': f"{output_dir}/{dataset_name}/train.no_label.csv",
        'test_split_with_labels': f"{output_dir}/{dataset_name}/test.csv",
        'test_split_no_labels': f"{output_dir}/{dataset_name}/test.no_label.csv",
        'dataset_knowledge': f"{output_dir}/{dataset_name}/dataset_description.md",
        'label_to_scalar': label_map,
        'class_col': label_col,
        'numeric_label_col': 'numeric_label'
    }
    
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=4))

def setup_agent_datasets(dataset_dir, dataset_agent_dir):
    dataset_agent_dir.mkdir(parents=True, exist_ok=True)

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

if __name__ == '__main__':
    args = parse_args()
    prepare_dataset(
        train=args.train,
        test=args.test,
        class_col=args.class_col,
        description=args.description,
        name=args.name,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        output_dir=args.output_dir
    )
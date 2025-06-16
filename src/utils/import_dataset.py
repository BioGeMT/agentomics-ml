#!/usr/bin/env python3
import argparse
import json
import pandas as pd
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Import dataset")
    parser.add_argument('--train', required=True, type=Path, help='Path to training file')
    parser.add_argument('--test', type=Path, help='Path to test file')
    parser.add_argument('--class', required=True, help='Class column name')
    parser.add_argument('--description', required=True, type=Path, help='Path to dataset description file')
    parser.add_argument('--positive-class', help='For binary classification: which class is positive (1)')
    parser.add_argument('--negative-class', help='For binary classification: which class is negative (0)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test) if args.test else None
    
    label_col = getattr(args, 'class')
    
    all_labels = train[label_col].dropna()
    if test is not None:
        all_labels = pd.concat([all_labels, test[label_col].dropna()])
    labels = list(all_labels.unique())
    
    if len(labels) == 2 and args.positive_class and args.negative_class:
        # binary classification 
        label_map = {args.negative_class: 0, args.positive_class: 1}
        print(f"Binary mapping: {args.negative_class}=0, {args.positive_class}=1")
    else:
        # multiclass: alphabetical order
        label_map = {lbl: i for i, lbl in enumerate(sorted(labels, key=str))}
        print(f"Label mapping: {label_map}")
    
    dataframes = [('train', train)]
    if test is not None:
        dataframes.append(('test', test))
    
    dataset_name = args.train.parent.name
    out_dir = Path('datasets') / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in dataframes:

        df['numeric_label'] = df[label_col].map(label_map)
        
        df.to_csv(out_dir / f'{name}.csv', index=False)
        
        df.drop([label_col, 'numeric_label'], axis=1).to_csv(
            out_dir / f'{name}.no_label.csv', index=False
        )
    
    description_content = args.description.read_text()
    (out_dir / 'dataset_description.md').write_text(description_content)
    
    meta = {
        'train_split': f"/repository/datasets/{dataset_name}/train.csv",
        'train_split_no_labels': f"/repository/datasets/{dataset_name}/train.no_label.csv",
        'test_split_with_labels': f"/repository/datasets/{dataset_name}/test.csv",
        'test_split_no_labels': f"/repository/datasets/{dataset_name}/test.no_label.csv",
        'dataset_knowledge': f"/repository/datasets/{dataset_name}/dataset_description.md",
        'label_to_scalar': label_map,
        'class_col': label_col,
        'numeric_label_col': 'numeric_label'
    }
    
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=4))

if __name__ == '__main__':
    main()
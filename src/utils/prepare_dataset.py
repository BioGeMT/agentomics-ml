import argparse
from pathlib import Path
from dataset_utils import prepare_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Prepares dataset for agent training and evaluation")
    parser.add_argument('--dataset-dir', required=True, type=Path, help='Directory containing the dataset files (train.csv, test.csv, etc.)')
    parser.add_argument('--target-col', dest='target_col', help='Name of the target column', type=str, default='target')
    parser.add_argument('--positive-class', help='Value used in the label column for a positive class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--negative-class', help='Value used in the label column for a negative class (affects some binary classification metrics). If not provided, numeric labels are assigned based on the label appearance order in the train csv file.', default=None)
    parser.add_argument('--task-type', choices=['classification', 'regression'], required=True, help='Type of machine learning task to prepare the dataset for.')
    parser.add_argument('--output-dir', type=Path, default=Path('../prepared_datasets').resolve(), help='Directory to save the processed dataset')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train = args.dataset_dir / 'train.csv'
    test = args.dataset_dir / 'test.csv' if (args.dataset_dir / 'test.csv').exists() else None
    description = args.dataset_dir / 'dataset_description.md' if (args.dataset_dir / 'dataset_description.md').exists() else None
    name = args.dataset_dir.name
    prepare_dataset(
        train=train,
        test=test,
        target_col=args.target_col,
        description=description,
        name=name,
        positive_class=args.positive_class,
        negative_class=args.negative_class,
        task_type=args.task_type,
        output_dir=args.output_dir
    )
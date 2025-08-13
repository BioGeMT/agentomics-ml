import sys
from pathlib import Path
import pandas as pd

def get_regression_candidates(dataset_dir: str) -> list:
    """
    Get regression candidates from a dataset directory.
    
    Args:
        dataset_dir: Path to dataset directory containing train.csv
        
    Returns:
        List of tuples (column_name, unique_count) for numeric columns with >10 unique values
    """
    dataset_path = Path(dataset_dir)
    train_path = dataset_path / 'train.csv'
    
    if not train_path.exists():
        return []
    
    try:
        df = pd.read_csv(train_path)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        candidates = []
        for col in numeric_cols:
            unique_count = df[col].nunique()
            if unique_count > 10:
                candidates.append((col, unique_count))
        
        return candidates
    except Exception:
        return []

def print_candidates_plain(candidates: list):
    """Print candidates in plain format (name|unique_count) for shell parsing."""
    for col_name, unique_count in candidates:
        print(f'{col_name}|{unique_count}')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_regression_candidates.py <dataset_dir>", file=sys.stderr)
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    candidates = get_regression_candidates(dataset_dir)
    print_candidates_plain(candidates)

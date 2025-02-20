import pandas as pd
from pathlib import Path
from genomic_benchmarks.loc2seq import download_dataset

def process_split(split, dataset_dir):
    data = []
    split_path = dataset_dir / split
    for class_name in ["negative", "positive"]:
        label = 0 if class_name == "negative" else 1
        class_dir = split_path / class_name
        for file_path in class_dir.iterdir():
            if file_path.is_file():
                # Read the sequence from the file and remove extra whitespace
                seq = file_path.read_text().strip()
                data.append({"sequence": seq, "class": label})
    return pd.DataFrame(data)

def main():
    dataset_dir = download_dataset("human_nontata_promoters", version=0)
    train_df = process_split("train", dataset_dir)
    test_df = process_split("test", dataset_dir)
    train_df.to_csv("human_nontata_promoters_train.csv", index=False)
    test_df.to_csv("human_nontata_promoters_test.csv", index=False)

if __name__ == "__main__":
    main()

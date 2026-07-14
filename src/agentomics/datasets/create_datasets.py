from pathlib import Path
import gzip
import json
import os
import shutil
import argparse
import urllib.request
import zipfile

from agentomics.datasets.data_contract import (
    DATASET_DESCRIPTION_FILE_NAME,
    ID_COLUMN_NAME,
    INPUT_DIR_NAME,
    LABEL_COLUMN_NAME,
    LABELS_FILE_NAME,
    METADATA_FILE_NAME,
    TEST_SPLIT,
    TRAIN_SPLIT,
)

REPO_PATH = Path.cwd()
FREE_SPOKEN_DIGIT_DATASET_VERSION = "v1.0.10"
MIRBENCH_ZENODO_RECORD_ID = "20540907"

MIRBENCH_DATASETS = {
    "AGO2_CLASH_Hejret2023": {
        "splits": ["train", "test"],
        "description": "The AGO2 Hejret2023 dataset was adapted from [miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA Frequency Class Bias]. This dataset contains microRNA sequences and their corresponding binding sites, as identified via a CLASH (crosslinking, ligation, and sequencing of hybrids) experiment. There are two sequences in this dataset: gene and noncodingRNA. The gene sequences are 50nt fragments including a target site of the noncodingRNA. We expect that the targeting occurs via partial complementarity of the two sequences. Samples with label==1 are target sites retrieved from the CLASH experiment. For each of these positive samples, a negative sample (label==0) is created by matching the same noncodingRNA sequence with a randomly selected gene sequence.",
    }
}

# Description pulled from the genomic benchmarks publication text
GENOMIC_BENCHMARKS_DATASETS = {
    "human_enhancers_cohn": "The Human enhancers Cohn dataset was adapted from [BioRxiv. 2018:264200]. Enhancers are genomic regulatory functional elements that can be bound by specific DNA binding proteins so as to regulate the transcription of a particular gene. Unlike promoters, enhancers do not need to be in a close proximity to the affected gene, and may be up to several million bases away, making their detection a difficult task.",
    "drosophila_enhancers_stark": "The Drosophila enhancers Stark dataset was adapted from [Nature. 2014;512(7512):91–5]. These enhancers were experimentally validated and we excluded the weak ones. Original coordinates referred to the dm3 [2007;316(5831):1625–8] assembly of the D. melanogaster genome. We used pyliftoverFootnote 3 tool to map coordinates to the dm6 assembly [Nucleic Acids Res. 2015;43(D1):690–7]. Negative sequences are randomly generated from drosophila genome dm6 to match lengths of positive sequences and to not overlap them.",
    "human_enhancers_ensembl": "The Human enhancers Ensembl dataset was constructed from Human enhancers from The FANTOM5 project [Nature. 2014;507(7493):455–61] accessed through the Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. Negative sequences have been randomly generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
    "human_nontata_promoters": "The Human non-TATA promoters dataset was adapted from [PLoS ONE. 2017;12(2):0171410]. These sequences are of length 251bp: from -200 to +50bp around transcription start site (TSS). To create non-promoters sequences of length 251bp, the authors of the original paper used random fragments of human genes located after first exons.",
    "human_ocr_ensembl": "The Human ocr Ensembl dataset was constructed from the Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. Positive sequences are Human Open Chromatin Regions (OCRs) from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1–8]. Open chromatin regions are regions of the genome that can be preferentially accessed by DNA regulatory elements because of their open chromatin structure. In the Ensembl Regulatory Build, this label is assigned to open chromatin regions, which were experimentally observed through DNase-seq, but covered by none of the other annotations (enhancer, promoter, gene, TSS, CTCF, etc.). Negative sequences were generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
    "human_ensembl_regulatory": "The Human regulatory Ensembl dataset was constructed from Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. This dataset has three classes: enhancer, promoter and open chromatin region from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1–8]."
}

LABEL_COLUMN = "target"

def _reset_dataset_dir(dataset_name: str) -> Path:
    local_dset_path = REPO_PATH / "datasets" / dataset_name
    if local_dset_path.exists():
        shutil.rmtree(local_dset_path)
    local_dset_path.mkdir(parents=True)
    return local_dset_path

def _write_metadata(dataset_dir: Path, task_type: str) -> None:
    (dataset_dir / METADATA_FILE_NAME).write_text(
        json.dumps({"task_type": task_type}, indent=4),
        encoding="utf-8",
    )

def _write_description(dataset_dir: Path, description: str) -> None:
    (dataset_dir / DATASET_DESCRIPTION_FILE_NAME).write_text(description, encoding="utf-8")

def _write_labels(split_dir: Path, rows: list[dict[str, str]]) -> None:
    import pandas as pd

    pd.DataFrame(rows, columns=[ID_COLUMN_NAME, LABEL_COLUMN_NAME]).to_csv(
        split_dir / LABELS_FILE_NAME,
        index=False,
    )

def _make_split_dirs(dataset_dir: Path, split_name: str, input_subdir: str | None = None) -> Path:
    split_dir = dataset_dir / split_name
    input_dir = split_dir / INPUT_DIR_NAME
    if input_subdir:
        input_dir = input_dir / input_subdir
    input_dir.mkdir(parents=True, exist_ok=True)
    return input_dir

def generate_mirbench_files(dataset: str | None = None):
    import pandas as pd

    from agentomics.datasets.csv_converter import convert_csv_dataset

    names = [dataset] if dataset is not None else list(MIRBENCH_DATASETS.keys())

    for dataset_name in names:
        info = MIRBENCH_DATASETS[dataset_name]
        local_dset_path = REPO_PATH / "datasets" / dataset_name
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(info["description"])

        split_frames = {}
        for split in info["splits"]:
            download_path = REPO_PATH / ".miRBench"
            os.makedirs(download_path, exist_ok=True)
            archive_path = download_path / f"{dataset_name}_{split}.tsv.gz"
            dataset_path = archive_path.with_suffix("")
            if not dataset_path.exists():
                if not archive_path.exists():
                    url = (
                        f"https://zenodo.org/records/{MIRBENCH_ZENODO_RECORD_ID}/files/"
                        f"{archive_path.name}?download=1"
                    )
                    urllib.request.urlretrieve(url, archive_path)
                with gzip.open(archive_path, "rb") as source:
                    with open(dataset_path, "wb") as destination:
                        shutil.copyfileobj(source, destination)
            df = pd.read_csv(dataset_path, sep="\t")
            df = df.rename(columns={"label": LABEL_COLUMN})
            split_frames[split] = df

        convert_csv_dataset(
            local_dset_path,
            label_column=LABEL_COLUMN,
            splits=split_frames,
            task_type="classification",
        )

def generate_genomic_benchmarks_files(dataset: str | None = None):
    import pandas as pd
    from genomic_benchmarks.loc2seq import download_dataset

    from agentomics.datasets.csv_converter import convert_csv_dataset

    names = [dataset] if dataset is not None else list(GENOMIC_BENCHMARKS_DATASETS.keys())

    for dataset_name in names:
        download_path = download_dataset(
            dataset_name,
            version=0,
            dest_path=REPO_PATH / ".genomic_benchmarks",
            cache_path=REPO_PATH / ".genomic_benchmarks",
        )

        local_dset_path = REPO_PATH / "datasets" / dataset_name
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(GENOMIC_BENCHMARKS_DATASETS[dataset_name])

        split_frames = {}
        for split in ["test", "train"]:
            data = []
            for label_path in (download_path / split).iterdir():
                label = label_path.stem
                for sequence_file in label_path.iterdir():
                    seq = sequence_file.read_text().strip()
                    data.append({"sequence": seq, LABEL_COLUMN: label})
            df = pd.DataFrame(data)
            split_frames[split] = df

        convert_csv_dataset(
            local_dset_path,
            label_column=LABEL_COLUMN,
            splits=split_frames,
            task_type="classification",
        )

def generate_breast_cancer_files(_dataset_name: str = "breast_cancer") -> None:
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from agentomics.datasets.csv_converter import convert_csv_dataset

    dataset_dir = _reset_dataset_dir(_dataset_name)
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": LABEL_COLUMN})
    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(lambda idx: data.target_names[int(idx)])
    df[ID_COLUMN_NAME] = [f"sample_{idx:04d}" for idx in range(len(df))]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COLUMN],
    )

    convert_csv_dataset(
        dataset_dir,
        label_column=LABEL_COLUMN,
        splits={TRAIN_SPLIT: train_df, TEST_SPLIT: test_df},
        id_column=ID_COLUMN_NAME,
        task_type="classification",
    )
    _write_description(
        dataset_dir,
        "# Breast Cancer Wisconsin Diagnostic\n\n"
        "Tabular binary classification dataset loaded with `sklearn.datasets.load_breast_cancer`.\n\n"
        "`input/data.csv` contains one row per sample with 30 real-valued features. "
        "`labels.csv` maps each `id` to the diagnosis label, where `malignant` and `benign` "
        "come from the scikit-learn target names.",
    )

def generate_digits_images_files(_dataset_name: str = "digits_images") -> None:
    import numpy as np
    from PIL import Image
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    dataset_dir = _reset_dataset_dir(_dataset_name)
    digits = load_digits()
    indices = np.arange(len(digits.target))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=digits.target,
    )

    split_indices = {
        TRAIN_SPLIT: train_indices,
        TEST_SPLIT: test_indices,
    }

    for split_name, split_idx in split_indices.items():
        images_dir = _make_split_dirs(dataset_dir, split_name, "images")
        labels = []
        for idx in sorted(split_idx):
            sample_id = f"digit_{idx:04d}"
            image = (digits.images[idx] / 16 * 255).astype(np.uint8)
            Image.fromarray(image, mode="L").save(images_dir / f"{sample_id}.png")
            labels.append({ID_COLUMN_NAME: sample_id, LABEL_COLUMN_NAME: str(digits.target[idx])})
        _write_labels(dataset_dir / split_name, labels)

    _write_metadata(dataset_dir, "classification")
    _write_description(
        dataset_dir,
        "# Digits Images\n\n"
        "Image classification dataset loaded with `sklearn.datasets.load_digits`.\n\n"
        "`input/images/` contains one PNG file per 8x8 handwritten digit image. "
        "Each row in `labels.csv` uses the image filename stem as `id` and the spoken/written "
        "digit class as `label`.",
    )

def generate_spoken_digits_files(_dataset_name: str = "spoken_digits") -> None:
    dataset_dir = _reset_dataset_dir(_dataset_name)
    download_dir = REPO_PATH / ".free_spoken_digit_dataset"
    download_dir.mkdir(exist_ok=True)
    archive_path = download_dir / f"free-spoken-digit-dataset-{FREE_SPOKEN_DIGIT_DATASET_VERSION}.zip"
    source_url = (
        "https://github.com/Jakobovski/free-spoken-digit-dataset/"
        f"archive/refs/tags/{FREE_SPOKEN_DIGIT_DATASET_VERSION}.zip"
    )

    if not archive_path.exists():
        temp_archive_path = archive_path.with_suffix(".zip.tmp")
        urllib.request.urlretrieve(source_url, temp_archive_path)
        temp_archive_path.replace(archive_path)

    recordings_dirs = sorted(download_dir.glob("free-spoken-digit-dataset-*/recordings"))
    if not recordings_dirs:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(download_dir)
        recordings_dirs = sorted(download_dir.glob("free-spoken-digit-dataset-*/recordings"))
    if not recordings_dirs:
        raise FileNotFoundError(f"No recordings directory found after extracting {archive_path}")

    recordings_dir = recordings_dirs[-1]
    split_labels = {TRAIN_SPLIT: [], TEST_SPLIT: []}
    for wav_path in sorted(recordings_dir.glob("*.wav")):
        label, _speaker, index = wav_path.stem.rsplit("_", 2)
        split_name = TEST_SPLIT if int(index) < 5 else TRAIN_SPLIT
        audio_dir = _make_split_dirs(dataset_dir, split_name, "audio")
        destination = audio_dir / wav_path.name
        shutil.copy2(wav_path, destination)
        split_labels[split_name].append({
            ID_COLUMN_NAME: wav_path.stem,
            LABEL_COLUMN_NAME: label,
        })

    for split_name, labels in split_labels.items():
        _write_labels(dataset_dir / split_name, labels)

    _write_metadata(dataset_dir, "classification")
    _write_description(
        dataset_dir,
        "# Spoken Digits\n\n"
        "Audio classification dataset generated from the Free Spoken Digit Dataset "
        f"({FREE_SPOKEN_DIGIT_DATASET_VERSION}).\n\n"
        "`input/audio/` contains one mono 8 kHz WAV file per spoken digit. "
        "Each row in `labels.csv` uses the WAV filename stem as `id` and the digit as `label`. "
        "The test split follows the dataset README convention: recording indices 0 through 4 "
        "are held out for test, while indices 5 through 49 are used for training.",
    )

DATASET_GENERATORS = {
    **{dataset_name: generate_genomic_benchmarks_files for dataset_name in GENOMIC_BENCHMARKS_DATASETS},
    **{dataset_name: generate_mirbench_files for dataset_name in MIRBENCH_DATASETS},
    "breast_cancer": generate_breast_cancer_files,
    "digits_images": generate_digits_images_files,
    "spoken_digits": generate_spoken_digits_files,
}

def list_dataset_names() -> None:
    print("Available example datasets:")
    for dataset_name in sorted(DATASET_GENERATORS):
        print(f"- {dataset_name}")

def generate_dataset_files(dataset: str, download_all: bool = False):
    if download_all:
        for dataset_name in sorted(DATASET_GENERATORS):
            DATASET_GENERATORS[dataset_name](dataset_name)
        return

    if dataset not in DATASET_GENERATORS:
        raise ValueError(f"Unknown dataset: {dataset}")
    DATASET_GENERATORS[dataset](dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset downloader helper script parser")
    parser.add_argument("--all", action="store_true", help="Download all built-in, miRBench and GenomicBenchmarks datasets (overrides --dataset)")
    parser.add_argument("--list", action="store_true", help="List available dataset names and exit")
    parser.add_argument(
        "--dataset", 
        default="AGO2_CLASH_Hejret2023",
        choices=sorted(DATASET_GENERATORS),
        help="Specific single dataset to download. Defaults to AGO2_CLASH_Hejret2023"
    )
    args = parser.parse_args()

    if args.list:
        list_dataset_names()
        raise SystemExit(0)

    generate_dataset_files(args.dataset, args.all)

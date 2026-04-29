import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from agentomics.utils.path_defaults import resolve_agentomics_paths


def _missing_dependency_message(package_name: str) -> str:
    return (
        f"Missing optional dependency '{package_name}'. "
        "Install Agentomics with the 'datasets' extra or use "
        "'agentomics-download-datasets' / './download_example_datasets.sh'."
    )


def _has_genomic_benchmark_split(dataset_path: Path, split: str) -> bool:
    split_path = dataset_path / split
    if not split_path.is_dir():
        return False

    for label_path in split_path.iterdir():
        if label_path.is_dir() and any(label_path.iterdir()):
            return True
    return False


def _is_complete_genomic_benchmark_cache(dataset_path: Path) -> bool:
    return all(_has_genomic_benchmark_split(dataset_path, split) for split in ("train", "test"))


def _download_genomic_benchmark_to_root(download_dataset, cache_root: Path, dataset_name: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    return download_dataset(
        dataset_name,
        dest_path=cache_root,
        cache_path=cache_root,
    )


def _download_genomic_benchmark_dataset(download_dataset, cache_root: Path, dataset_name: str) -> Path:
    dataset_cache_path = cache_root / dataset_name
    if _is_complete_genomic_benchmark_cache(dataset_cache_path):
        return dataset_cache_path

    try:
        return _download_genomic_benchmark_to_root(download_dataset, cache_root, dataset_name)
    except OSError as exc:
        if exc.errno not in {16, 39}:
            raise

        fresh_root = Path(tempfile.mkdtemp(prefix=f"{dataset_name}-", dir=cache_root.parent))
        return _download_genomic_benchmark_to_root(download_dataset, fresh_root, dataset_name)


def generate_mirbench_files(datasets_dir: Path, cache_dir: Path) -> None:
    try:
        from miRBench.dataset import download_dataset as mirbench_download_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "miRBench":
            raise SystemExit(_missing_dependency_message("miRBench")) from exc
        raise

    dataset_names_splits = {
        "AGO2_CLASH_Hejret2023": ["train", "test"],
    }

    ago2_clash_description = """
        The AGO2 Hejret2023 dataset was adapted from [miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA Frequency Class Bias].
        This dataset contains microRNA sequences and their corresponding binding sites, as
        identified via a CLASH (crosslinking, ligation, and sequencing of hybrids) experiment.
        There are two sequences in this dataset: gene and noncodingRNA.
        The gene sequences are 50nt fragments including a target site of the noncodingRNA.
        We expect that the targeting occurs via partial complementarity of the two sequences.
        Samples with label==1 are target sites retrieved from the CLASH experiment.
        For each of these positive samples, a negative sample (label==0) is created by matching the same
        noncodingRNA sequence with a randomly selected gene sequence.
    """
    dataset_description = {
        "AGO2_CLASH_Hejret2023": ago2_clash_description,
    }
    class_col = "target"

    for dataset_name in dataset_names_splits:
        local_dataset_path = datasets_dir / dataset_name
        local_dataset_path.mkdir(parents=True, exist_ok=True)
        (local_dataset_path / "dataset_description.md").write_text(
            dataset_description[dataset_name],
            encoding="utf-8",
        )

    for dataset_name, splits in dataset_names_splits.items():
        local_dataset_path = datasets_dir / dataset_name
        download_path = cache_dir / ".miRBench" / dataset_name
        download_path.mkdir(parents=True, exist_ok=True)
        for split in splits:
            split_path = download_path / f"{split}.tsv"
            mirbench_download_dataset(dataset_name, download_path=split_path, split=split)
            df = pd.read_csv(split_path, sep="\t")
            df = df.rename(columns={"label": class_col})
            # Keep original target column; numeric labels are created during preparation.
            df.to_csv(local_dataset_path / f"{split}.csv", index=False)


def generate_genomic_benchmarks_files(datasets_dir: Path, cache_dir: Path) -> None:
    try:
        from genomic_benchmarks.loc2seq import download_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "genomic_benchmarks":
            raise SystemExit(_missing_dependency_message("genomic-benchmarks")) from exc
        raise

    dataset_description = {
        "human_enhancers_cohn": "The Human enhancers Cohn dataset was adapted from [BioRxiv. 2018:264200]. Enhancers are genomic regulatory functional elements that can be bound by specific DNA binding proteins so as to regulate the transcription of a particular gene. Unlike promoters, enhancers do not need to be in a close proximity to the affected gene, and may be up to several million bases away, making their detection a difficult task.",
        "drosophila_enhancers_stark": "The Drosophila enhancers Stark dataset was adapted from [Nature. 2014;512(7512):91-5]. These enhancers were experimentally validated and we excluded the weak ones. Original coordinates referred to the dm3 [2007;316(5831):1625-8] assembly of the D. melanogaster genome. We used pyliftoverFootnote 3 tool to map coordinates to the dm6 assembly [Nucleic Acids Res. 2015;43(D1):690-7]. Negative sequences are randomly generated from drosophila genome dm6 to match lengths of positive sequences and to not overlap them.",
        "human_enhancers_ensembl": "The Human enhancers Ensembl dataset was constructed from Human enhancers from The FANTOM5 project [Nature. 2014;507(7493):455-61] accessed through the Ensembl database [Nucleic Acids Res. 2021;49(D1):884-91]. Negative sequences have been randomly generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
        "human_nontata_promoters": "The Human non-TATA promoters dataset was adapted from [PLoS ONE. 2017;12(2):0171410]. These sequences are of length 251bp: from -200 to +50bp around transcription start site (TSS). To create non-promoters sequences of length 251bp, the authors of the original paper used random fragments of human genes located after first exons.",
        "human_ocr_ensembl": "The Human ocr Ensembl dataset was constructed from the Ensembl database [Nucleic Acids Res. 2021;49(D1):884-91]. Positive sequences are Human Open Chromatin Regions (OCRs) from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1-8]. Open chromatin regions are regions of the genome that can be preferentially accessed by DNA regulatory elements because of their open chromatin structure. In the Ensembl Regulatory Build, this label is assigned to open chromatin regions, which were experimentally observed through DNase-seq, but covered by none of the other annotations (enhancer, promoter, gene, TSS, CTCF, etc.). Negative sequences were generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
        "human_ensembl_regulatory": "The Human regulatory Ensembl dataset was constructed from Ensembl database [Nucleic Acids Res. 2021;49(D1):884-91]. This dataset has three classes: enhancer, promoter and open chromatin region from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1-8].",
    }

    for dataset_name in dataset_description:
        cache_root = cache_dir / ".genomic_benchmarks"
        cache_root.mkdir(parents=True, exist_ok=True)
        download_path = _download_genomic_benchmark_dataset(download_dataset, cache_root, dataset_name)

        local_dataset_path = datasets_dir / dataset_name
        class_col = "target"
        local_dataset_path.mkdir(parents=True, exist_ok=True)
        (local_dataset_path / "dataset_description.md").write_text(
            dataset_description[dataset_name],
            encoding="utf-8",
        )

        for split in ["test", "train"]:
            data = []
            for label_path in (download_path / split).iterdir():
                label = label_path.stem
                for sequence_file in label_path.iterdir():
                    seq = sequence_file.read_text().strip()
                    data.append({"sequence": seq, class_col: label})
            df = pd.DataFrame(data)
            # Keep original target column; numeric labels are created during preparation.
            df.to_csv(local_dataset_path / f"{split}.csv", index=False)


def generate_dataset_files(datasets_dir: Path, cache_dir: Path) -> None:
    datasets_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    generate_genomic_benchmarks_files(datasets_dir, cache_dir)
    generate_mirbench_files(datasets_dir, cache_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Download example datasets for Agentomics.")
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        help="Directory where datasets will be written. Defaults to DATASETS_DIR or ./datasets from the repo root/current working directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory used to cache downloaded source datasets. Defaults to the repo root/current working directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = resolve_agentomics_paths(datasets_dir=args.datasets_dir)
    cache_dir = args.cache_dir.resolve() if args.cache_dir is not None else paths.base_dir
    generate_dataset_files(paths.datasets_dir, cache_dir)


if __name__ == "__main__":
    main()

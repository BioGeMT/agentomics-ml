from pathlib import Path
import os
import pandas as pd
import argparse

from datasets.csv_converter import convert_csv_dataset

REPO_PATH = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent

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

def generate_mirbench_files(dataset: str | None = None):
    from miRBench.dataset import download_dataset as mirbench_download_dataset

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
            mirbench_download_dataset(dataset_name, download_path=download_path / 'miRBench', split=split)
            df = pd.read_csv(download_path / 'miRBench', sep="\t")
            df = df.rename(columns={"label": LABEL_COLUMN})
            split_frames[split] = df

        convert_csv_dataset(
            local_dset_path, label_column=LABEL_COLUMN, splits=split_frames, task_type="classification",
        )

def generate_genomic_benchmarks_files(dataset: str | None = None):
    from genomic_benchmarks.loc2seq import download_dataset

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
            local_dset_path, label_column=LABEL_COLUMN, splits=split_frames, task_type="classification",
        )

def generate_dataset_files(dataset: str, download_all: bool = False):
    if download_all:
        generate_genomic_benchmarks_files()
        generate_mirbench_files()
    else:
        if dataset in GENOMIC_BENCHMARKS_DATASETS:
            generate_genomic_benchmarks_files(dataset)
        elif dataset in MIRBENCH_DATASETS:
            generate_mirbench_files(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset downloader helper script parser")
    parser.add_argument("--all", action="store_true", help="Download all datasets from the miRBench and GenomicBenchmarks collections (overrides --dataset)")
    parser.add_argument(
        "--dataset", 
        default="AGO2_CLASH_Hejret2023",
        choices=GENOMIC_BENCHMARKS_DATASETS.keys() | MIRBENCH_DATASETS.keys(),
        help="Specific single dataset to download. Defaults to AGO2_CLASH_Hejret2023"
    )
    args = parser.parse_args()

    generate_dataset_files(args.dataset, args.all)

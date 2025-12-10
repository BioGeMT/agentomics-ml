from pathlib import Path
import os
import json
import pandas as pd

def generate_mirbench_files():
    from miRBench.dataset import download_dataset as mirbench_download_dataset
    repo_path = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
    dataset_names_splits = {
        "AGO2_CLASH_Hejret2023": ["train", "test"],
    }

    ago2_clash_description = "The AGO2 Hejret2023 dataset was adapted from miRBench, which catalogues microRNA binding sites measured with CLASH (crosslinking, ligation, and sequencing of hybrids). Each example pairs a 50 nt gene fragment containing the candidate binding site with a 17–26 nt microRNA sequence. Positive labels correspond to miRNA–mRNA hybrids observed in CLASH, whereas negatives reuse the microRNA but couple it with a randomly selected gene fragment. The CSV files store columns gene, noncodingRNA, noncodingRNA_name, noncodingRNA_fam, feature, target, chr, start, end, strand, and gene_cluster_ID, where target is 1 for true interactions and 0 otherwise. The combined train and test partitions contain 4,579 positives and 4,579 negatives, yielding a balanced benchmark for microRNA target recognition."
    dataset_desrciption = {
        "AGO2_CLASH_Hejret2023": ago2_clash_description,
    }
    dataset_label_to_scalar = {
        "AGO2_CLASH_Hejret2023": {1:1, 0:0},
    }
    class_col = "target"
    for dataset_name in dataset_names_splits.keys():
        local_dset_path = repo_path / "datasets" / dataset_name
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(dataset_desrciption[dataset_name])

    for dataset_name, splits in dataset_names_splits.items():
        for split in splits:
            download_path = repo_path/".miRBench"
            os.makedirs(download_path, exist_ok=True)
            mirbench_download_dataset(dataset_name, download_path=download_path/'miRBench', split=split)
            df = pd.read_csv(download_path/'miRBench', sep="\t")
            df = df.rename(columns={"label": class_col})
            # Keep original target column - 'numeric_label' will be created during preparation
            df.to_csv(f"{local_dset_path}/{split}.csv", index=False)

def generate_genomic_benchmarks_files():
    from genomic_benchmarks.loc2seq import download_dataset
    # Description pulled from the genomic benchmarks publication text
    dataset_description = {
        "human_enhancers_cohn": "The Human enhancers Cohn dataset follows the benchmark created by Cohn et al. (2018) for enhancer identification. Positive samples are 500 bp human enhancer loci supplied by the authors and remapped to the GRCh38 reference genome, while negatives are length-matched genomic segments without enhancer annotations. Each CSV row holds sequence and target columns, where target is positive or negative. The dataset remains nearly balanced with 13,895 positives and 13,896 negatives across train and test splits, providing a curated binary task for evaluating enhancer detection models on human DNA.",
        "drosophila_enhancers_stark": "The Drosophila enhancers Stark dataset contrasts experimentally validated enhancer regions from Stark Lab with synthetic background loci sampled from the Drosophila melanogaster dm6 genome. Positive sequences were originally reported on the dm3 assembly and lifted over to dm6, with weak enhancer calls removed during curation. Negative sequences are random dm6 fragments matched in length to the positives and filtered to avoid overlap. The CSV files contain columns sequence and target, with target equal to positive or negative. Across train and test splits there are 3,457 samples per class, and sequences span roughly two kilobases on average.",
        "human_enhancers_ensembl": "The Human enhancers Ensembl dataset aggregates enhancer elements from the FANTOM5 atlas via the Ensembl Regulatory Build. Positive sequences cover enhancers active across diverse human cell and tissue types profiled with Cap Analysis of Gene Expression, and negatives are random GRCh38 loci length-matched to the enhancers and filtered to remove regulatory annotations. CSV files expose sequence and target columns, where target is positive or negative. The benchmark is class-balanced with 77,421 sequences per label across training and test splits, and typical sequence spans are a few hundred base pairs.",
        "human_nontata_promoters":"The Human non-TATA promoters dataset was adapted from [PLoS ONE. 2017;12(2):0171410]. These sequences are of length 251bp: from -200 to +50bp around transcription start site (TSS). To create non-promoters sequences of length 251bp, the authors of the original paper used random fragments of human genes located after first exons.",
        "human_ocr_ensembl": "The Human open chromatin Ensembl dataset collects open chromatin regions from the Ensembl Regulatory Build (release 100). Positive sequences correspond to DNase-seq accessible regions lacking more specific annotations such as promoters or enhancers; negatives are random GRCh38 segments matched in length and filtered to avoid regulatory overlaps. CSV files contain sequence and target columns, with target equal to positive or negative. The dataset is balanced with 87,378 sequences per class across training and test partitions, highlighting chromatin accessibility in few-hundred-base-pair windows.",
        "human_ensembl_regulatory":"The Human regulatory Ensembl dataset was constructed from Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. This dataset has three classes: enhancer, promoter and open chromatin region from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1–8].",
    }
    dataset_label_to_scalar = {
        "human_enhancers_cohn": {"positive": 1, "negative": 0},
        "drosophila_enhancers_stark": {"positive": 1, "negative": 0},
        "human_enhancers_ensembl": {"positive": 1, "negative": 0},
        "human_nontata_promoters": {"positive": 1, "negative": 0},
        "human_ocr_ensembl": {"positive": 1, "negative": 0},
        "human_ensembl_regulatory": {"ocr": 1, "enhancer": 0, "promoter": 2},
    }

    for dataset_name in dataset_description.keys():
        repo_path = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent

        download_path = download_dataset(dataset_name, dest_path=repo_path/".genomic_benchmarks", cache_path=repo_path/".genomic_benchmarks")

        local_dset_path = repo_path / "datasets" / dataset_name
        class_col = "target"
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(dataset_description[dataset_name])

        for split in ["test","train"]:
            data = []
            for label_path in (download_path/split).iterdir():
                label = label_path.stem
                for sequence_file in label_path.iterdir():
                    seq = sequence_file.read_text().strip()
                    data.append({"sequence": seq, class_col: label})
            df = pd.DataFrame(data)
            # Keep original target column - 'numeric_label' will be created during preparation
            df.to_csv(f"{local_dset_path}/{split}.csv", index=False)

def generate_dataset_files():
    generate_genomic_benchmarks_files()
    generate_mirbench_files()

if __name__ == "__main__":
    generate_dataset_files()
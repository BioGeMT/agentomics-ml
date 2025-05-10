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
    dataset_desrciption = {
        "AGO2_CLASH_Hejret2023": ago2_clash_description,
    }
    dataset_label_to_scalar = {
        "AGO2_CLASH_Hejret2023": {1:1, 0:0},
    }
    class_col = "label"
    numeric_label_col = "numeric_label"
    for dataset_name in dataset_names_splits.keys():
        local_dset_path = repo_path / "datasets" / dataset_name
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(dataset_desrciption[dataset_name])

        with open(f"{local_dset_path}/metadata.json", "w") as f:
            default_docker_path = f'/repository/datasets/{dataset_name}'
            metadata = {
                "train_split": f"{default_docker_path}/train.csv",
                "test_split_with_labels": f"{default_docker_path}/test.csv",
                "test_split_no_labels": f"{default_docker_path}/test.no_label.csv",
                "dataset_knowledge":f"{default_docker_path}/dataset_description.md",
                "label_to_scalar": dataset_label_to_scalar[dataset_name],
                "class_col": class_col,
                "numeric_label_col": numeric_label_col,
            }
            f.write(json.dumps(metadata, indent=4))

    for dataset_name, splits in dataset_names_splits.items():
        for split in splits:
            download_path = repo_path/".miRBench"
            os.makedirs(download_path, exist_ok=True)
            mirbench_download_dataset(dataset_name, download_path=download_path/'miRBench', split=split)
            df = pd.read_csv(download_path/'miRBench', sep="\t")
            df[numeric_label_col] = df[class_col].map(dataset_label_to_scalar[dataset_name])
            df.to_csv(f"{local_dset_path}/{split}.csv", index=False)
            if(split == "test"):
                df.drop(columns=[class_col, numeric_label_col]).to_csv(f"{local_dset_path}/{split}.no_label.csv", index=False)

def generate_genomic_benchmarks_files():
    from genomic_benchmarks.loc2seq import download_dataset
    # Description pulled from the genomic benchmarks publication text
    dataset_description = {
        "human_enhancers_cohn": "The Human enhancers Cohn dataset was adapted from [BioRxiv. 2018:264200]. Enhancers are genomic regulatory functional elements that can be bound by specific DNA binding proteins so as to regulate the transcription of a particular gene. Unlike promoters, enhancers do not need to be in a close proximity to the affected gene, and may be up to several million bases away, making their detection a difficult task.",
        "drosophila_enhancers_stark": "The Drosophila enhancers Stark dataset was adapted from [Nature. 2014;512(7512):91–5]. These enhancers were experimentally validated and we excluded the weak ones. Original coordinates referred to the dm3 [2007;316(5831):1625–8] assembly of the D. melanogaster genome. We used pyliftoverFootnote 3 tool to map coordinates to the dm6 assembly [Nucleic Acids Res. 2015;43(D1):690–7]. Negative sequences are randomly generated from drosophila genome dm6 to match lengths of positive sequences and to not overlap them.",
        "human_enhancers_ensembl":"The Human enhancers Ensembl dataset was constructed from Human enhancers from The FANTOM5 project [Nature. 2014;507(7493):455–61] accessed through the Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. Negative sequences have been randomly generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
        "human_nontata_promoters":"The Human non-TATA promoters dataset was adapted from [PLoS ONE. 2017;12(2):0171410]. These sequences are of length 251bp: from -200 to +50bp around transcription start site (TSS). To create non-promoters sequences of length 251bp, the authors of the original paper used random fragments of human genes located after first exons.",
        "human_ocr_ensembl":"The Human ocr Ensembl dataset was constructed from the Ensembl database [Nucleic Acids Res. 2021;49(D1):884–91]. Positive sequences are Human Open Chromatin Regions (OCRs) from The Ensembl Regulatory Build [Genome Biol. 2015;16(1):1–8]. Open chromatin regions are regions of the genome that can be preferentially accessed by DNA regulatory elements because of their open chromatin structure. In the Ensembl Regulatory Build, this label is assigned to open chromatin regions, which were experimentally observed through DNase-seq, but covered by none of the other annotations (enhancer, promoter, gene, TSS, CTCF, etc.). Negative sequences were generated from the Human genome GRCh38 to match the lengths of positive sequences and not overlap them.",
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
        class_col = "class"
        numeric_label_col = "numeric_label"
        os.makedirs(local_dset_path, exist_ok=True)

        with open(f"{local_dset_path}/dataset_description.md", "w") as f:
            f.write(dataset_description[dataset_name])

        with open(f"{local_dset_path}/metadata.json", "w") as f:
            default_docker_path = f'/repository/datasets/{dataset_name}'
            metadata = {
                "train_split": f"{default_docker_path}/train.csv",
                "test_split_with_labels": f"{default_docker_path}/test.csv",
                "test_split_no_labels": f"{default_docker_path}/test.no_label.csv",
                "dataset_knowledge":f"{default_docker_path}/dataset_description.md",
                "label_to_scalar": dataset_label_to_scalar[dataset_name],
                "class_col": class_col,
                "numeric_label_col": numeric_label_col,
            }
            f.write(json.dumps(metadata, indent=4))

        for split in ["test","train"]:
            data = []
            for label_path in (download_path/split).iterdir():
                label = label_path.stem
                for sequence_file in label_path.iterdir():
                    seq = sequence_file.read_text().strip()
                    data.append({"sequence": seq, "class": label})
            df = pd.DataFrame(data)
            df[numeric_label_col] = df[class_col].map(dataset_label_to_scalar[dataset_name])
            df.to_csv(f"{local_dset_path}/{split}.csv", index=False)
            if(split == "test"):
                df.drop(columns=[class_col, numeric_label_col]).to_csv(f"{local_dset_path}/{split}.no_label.csv", index=False)

def generate_dataset_files():
    generate_genomic_benchmarks_files()
    generate_mirbench_files()

        The AGO2 Hejret2023 dataset was adapted from [miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA Frequency Class Bias]. 
        This dataset contains microRNA sequences and their corresponding binding sites, as 
        identified via a CLASH (crosslinking, ligation, and sequencing of hybrids) experiment. 
        There are two sequences in this dataset: gene and noncodingRNA. 
        The gene sequences are 50nt fragments including a target site of the noncodingRNA. 
        We expect that the targeting occurs via partial complementarity of the two sequences.
        Samples with label==1 are target sites retrieved from the CLASH experiment. 
        For each of these positive samples, a negative sample (label==0) is created by matching the same 
        noncodingRNA sequence with a randomly selected gene sequence.
    
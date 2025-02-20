# Dataset Information

**Name:** human_non_tata_promoters

Contains two columns, comma separated, with the following information:

_sequence_: DNA sequence to be classified {alphabet: A, C, G, T, N}

_class_: 1 if the promoter is a non-TATA promoter, 0 otherwise


## Extended Information

This is a collection of DNA sequences that encode for non-TATA promoters. The sequences are of length 1000. The dataset is balanced, with 50% of the sequences being non-TATA promoters and 50% being randomly generated sequences.

Umarov et al, show: Sensitivity = 0.90, Specificity = 0.98 for non-TATA promoter sequences.

Umarov et al:  Adam optimizer was used for training with categorical cross-entropy as a loss function. Our CNN architecture (Fig 1) in most cases consisted of just one convolutional layer with 200 filters having length 21. After convolutional layer, we have a standard Max-Pooling layer. The output from the Max-Pooling layer is fed into a standard fully connected ReLU layer with 128 neurons. Pooling size was usually 2. Finally, the ReLU layer is connected to output layer with sigmoid activation, where neurons correspond to promoter and non-promoter classes. The batch size used for training was 16. Input of the network consisted of nucleotide sequences where each nucleotide is encoded by a four dimensional vector A (1,0,0,0), T(0,1,0,0), G(0,0,1,0) and C(0,0,0,1). Output is a two dimensional vector: promoter (1, 0) and Non-promoter (0, 1) prediction. The training takes a few minutes on GTX 980 Ti GPU.  We intentionally used, in most cases, one layer CNN architecture, but sometimes to get a proper balance of accuracy between positives examples (promoters) and negative examples (non-promoter) two or three layers may be applied. 


Gresova et al, show: Accuracy = 86.5, F1 score = 84.4 for non-TATA promoter sequences.

Gresova et al:  CNN is an architecture that is able to find input features without feature engineering and has a relatively small number of parameters due to weights sharing (see [49] for more). Our implementation consists of three convolutional layers with 16, 8, and 4 filters, with a kernel size of 8. The output of each convolutional layer goes through the batch normalization layer and the max-pooling layer. The output of the last set of layers is flattened and goes through two dense layers. The last layer is designed to predict probabilities that the input sample belongs to any of the given classes. 


## Citation Source

Grešová, Katarína, et al. "Genomic benchmarks: a collection of datasets for genomic sequence classification." BMC Genomic Data 24.1 (2023): 25.

Umarov RK, Solovyev VV. Recognition of prokaryotic and eukaryotic promoters using convolutional deep learning neural networks. PLoS ONE. 2017;12(2):0171410.

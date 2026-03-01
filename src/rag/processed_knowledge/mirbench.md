<!-- image -->

<!-- image -->

## miRBench: novel benchmark datasets for microRNA binding site prediction that mitigate against prevalent microRNA frequency class bias

Stephanie Sammut 1,2,† , Katarina Gresova 1,2,3,† , Dimosthenis Tzimotoudis 1,2 , Eva Marsalkova 3,4 , David Cechak 3,4 , Panagiotis Alexiou 1,2, �

1 Centre for Molecular Medicine and Biobanking, University of Malta, Msida, MSD 2080, Malta

3 National Centre for Biomolecular Research, Faculty of Science, Masaryk University, Brno 61137, Czech Republic

2 Department of Applied Biomedical Science, Faculty of Health Sciences, University of Malta, Msida, MSD 2080, Malta

4 Central European Institute of Technology, Masaryk University, Brno, 62500, Czech Republic

� Corresponding author. Centre for Molecular Medicine and Biobanking, Room 412, Biomedical Sciences Building, University of Malta, Msida, MSD 2080, Malta. E-mail: panagiotis.alexiou@um.edu.mt.

† ¼ equal contribution.

## Abstract

Motivation: MicroRNAs (miRNAs) are crucial regulators of gene expression, but the precise mechanisms governing their binding to target sites remain unclear. A major contributing factor to this is the lack of unbiased experimental datasets for training accurate prediction models. While recent experimental advances have provided numerous miRNA-target interactions, these are solely positive interactions. Generating negative examples in silico is challenging and prone to introducing biases, such as the miRNA frequency class bias identified in this work. Biases within datasets can compromise model generalization, leading models to learn dataset-specific artifacts rather than true biological patterns.

Results: We introduce a novel methodology for negative sample generation that effectively mitigates the miRNA frequency class bias. Using this methodology, we curate several new, extensive datasets and benchmark several state-of-the-art methods on them. We find that a simple convolutional neural network model, retrained on some of these datasets, is able to outperform state-of-the-art methods reaching average precision scores between 0.81 and 0.86 in test datasets. This highlights the potential for leveraging unbiased datasets to achieve improved performance in miRNA binding site prediction. To facilitate further research and lower the barrier to entry for machine learning researchers, we provide an easily accessible Python package, miRBench, for dataset retrieval, sequence encoding, and the execution of state-of-the-art models.

Availability and implementation: The miRBench Python package is accessible at https://github.com/katarinagresova/miRBench/releases/tag/ v1.0.1.

## Graphical abstract

miRBench i543

## 1 Introduction

Over 30 years ago, the regulatory map of the animal cell was fundamentally  changed  when  fragments  of  cellular  RNA, previously  considered  nonfunctional,  were  identified  as  important  regulators  of  the  post-transcriptional  life  of  RNA (Lee et  al. 1993).  MicroRNAs  (miRNAs),  small  regulatory RNAs,  were  rapidly  found  to  serve  diverse  roles,  amongst which,  functioning  as  master  regulators  of  gene  expression during  embryogenesis,  tissue  development,  and  the  maintenance of homeostasis in adults (Bernstein et al. 2003, Zhao et al. 2005, Ivey and Srivastava 2010). Dysregulated miRNAs are  linked  to  various  diseases,  including  cancer  (He  and Hannon 2004, Calin and Croce 2006, Esquela-Kerscher and Slack 2006), cardiovascular diseases (van Rooij et al. 2006, Ikeda et al. 2007, Thum and Condorelli 2015), neurological disorders (H � ebert and De Strooper 2009), and immune conditions  (O'Connell et  al. 2007, Sonkoly and Pivarcsi 2009, Dai and Ahmed 2011). These molecules also hold promise as biomarkers (Condrat et al. 2020) and therapeutics (van Rooij and Olson 2012, Rupaimoole and Slack 2017, Chakraborty et al. 2017), including recent applications in CAR-T therapy (Rad et  al. 2022,  Shen et  al. 2024).  The  importance  of miRNAs  in  biology  and  biomedicine  was  highlighted  with the Nobel Prize in Medicine in 2024 for their discovery.

miRNAs  function  by  associating  with  proteins  from  the Argonaute  (AGO)  family,  which  are  fundamental  components  of  the  RNA-induced  silencing  complex  (RISC).  In mammals, the four AGO proteins (AGO1-4) specifically interact  with  miRNAs  to  form  ribonucleoprotein  complexes central to RNA silencing. Among them, AGO2 is especially crucial; unlike AGO1, AGO3, and AGO4, its absence results in embryonic lethality in mice, highlighting its indispensable role  (Liu et  al. 2004, Morita et  al. 2007). AGO2 identifies target RNA molecules by utilizing partial sequence complementarity  between  the  'guide'  miRNA  and  the  'target' RNA  (Bartel  2004).  In  mammals  most  known  AGO2 miRNA binding sites show partial complementarity, often focused on a 'seed' region located at the 5 0 end of the 'guide' sequence.  A  'canonical  seed'  sequence  denotes  a  fully Watson-Crick complementary stretch of at least six nucleotides  starting  at  the  second  position  from  the  5 0 end  of  the miRNA 'guide'. Further binding outside the seed area can stabilize the interaction and is commonly called 3 0 compensatory binding. However, functional interactions not mediated by a 'canonical seed' have been known since the early days of  miRNA  targeting  research  in  worms  and  mammals (Didiano and Hobert 2006, Broughton et al. 2016). We, and others, have previously shown that less than 50% of experimentally identified miRNA binding sites are mediated by a canonical seed (Helwak et al. 2013, Klimentov � a et al. 2022, Hejret et al. 2023). While the exact rules of AGO2: miRNAtarget  recognition  remain  largely  unknown,  computational prediction programs  have  been  developed  that  rely  on approximations  of  these  rules,  and  form  a  crucial  part  of miRNA-target gene prediction pipelines (Alexiou et al. 2009, Gre � sov � a et al. 2022). These programs commonly employ target site or 'miRNA binding site' prediction methods, which are not only vital components of larger pipelines but also essential tools for gaining deeper insights into the intricacies of AGO:miRNA:target interactions.

Several miRNA binding site classification models have been developed in the past few years with varying degrees of accuracy and generalizability (McGeary et al. 2019, Zheng et al. 2020,

Min et  al. 2022,  Klimentov � a et  al. 2022,  Hejret et  al. 2023, Yang et al. 2024). These models are based on various types of neural networks, including convolutional neural networks, residual networks, and attention networks as common architectures. Neural networks require substantial amounts of data to learn meaningful information related to the task at hand.

In  2013,  an  experimental method termed CLASH (Cross Linking, Ligation, and Sequencing of Hybrids) was developed (Helwak et  al. 2013),  utilizing  a  ligation  step  between  the 'guide' and 'target' sequences, produced 'chimeric' reads that contain both sequences of the AGO-bound guide:target interaction.  More  recently,  another  high-throughput  technique  termed  chimeric  eCLIP  (enhanced  Crosslinking  and Immunoprecipitation) was developed (Manakov et al. 2022). The method integrates a chimeric ligation step into the eCLIP method, which has an improved library generation efficiency relative to earlier CLIP methods (Van Nostrand et al. 2016). Chimeric eCLIP is therefore also able to capture AGO-bound chimeric reads of guide and target RNA sequences, but at an even  higher  resolution  and  sensitivity,  providing  a  comprehensive map of miRNA binding sites. An important distinction between the two methods is that in the CLASH method, AGO protein is induced, whereas chimeric eCLIP leverages endogenous AGO protein. Consequently, CLASH may be retrieving more interactions of lower affinity due to the higher abundance of AGO in CLASH compared to chimeric eCLIP, as we have previously suggested (Hejret et al. 2023).

The majority of miRNA binding site classification models are  trained  on  datasets  derived  from  experiments  that  produce  chimeric  miRNA-binding  site  interactions  (Helwak et al. 2013, Klimentov � a et al. 2022, Hejret et al. 2023), or on databases  of  experimentally  derived  binding  sites  (Vlachos et  al. 2015, Chou et  al. 2016, Pla et  al. 2018). One model (McGeary et al. 2019) is instead trained on RNA Bind-n-Seq data,  which  yields  a  continuous  estimation  of  dissociation constants  that  can  serve  as  a  proxy  for  binding  affinity. Importantly, while these datasets predominantly contain experimentally  validated  target  sites,  representing  'positive interactions', machine learning approaches require a comparable number of 'negative examples' for effective training. However, experimentally confirming miRNA non -binding sequence pairs is significantly more challenging. Consequently, negative  examples  are  often  underrepresented  in  databasederived datasets and are typically generated in silico for chimeric miRNA-binding site interaction datasets.

In  this  study,  we  uncover  what  we  term  a  miRNA  frequency  class  bias  in  existing  miRNA  binding  site  datasets used for training and benchmarking (Pla et al. 2018, Hejret et al. 2023). This bias occurs when the frequency distribution of miRNAs in the negative class differs from that in the positive class, arising from how negative examples are generated in silico . The use of such datasets for training on the task of miRNA binding site prediction results in models that struggle to generalize well, as they learn sequence interaction patterns muffled by the intricacies of the data.

To  mitigate  this  bias,  we  developed  a  novel  strategy  for generating  negative  examples.  We  applied  this  strategy  to existing datasets to create new, unbiased versions, and also curated  a  new  dataset  based  on  human  AGO2  chimeric miRNA binding site interactions from a recent experimental method (Manakov et al. 2022). This new dataset, processed using  our  negative  example  generation  strategy,  yielded  a training set containing over 2.5 million � 1:1 class-balanced

i544 Sammut et al.

miRNA-binding site interactions. Alongside this large training set, we also generated smaller benchmarking datasets, all carefully constructed to mitigate the miRNA frequency class bias. We demonstrate that models trained on these unbiased datasets, particularly the large Manakov2022-derived training set, generalize better and outperform the current state-ofthe-art on these benchmarks.

We make these new training and  benchmarking datasets publicly available and provide a user-friendly Python package for easy access to these datasets, as well as to the state-of-theart models benchmarked in this work.

## 2 Materials and methods

## 2.1 Identification of miRNA frequency class bias in published datasets

To investigate potential imbalances in miRNA frequency between  positive  and  negative  classes  in  widely  used  miRNA binding  site  interaction  datasets,  we  performed  a  series  of analyses using simple classification models. We hypothesized that  such  imbalances,  which  we  term  'miRNA  frequency class  bias',  could  lead  to  artificially  inflated  performance metrics for models trained on these datasets.

Two independent, publicly available datasets were used:

- a) Original\_Hejret2023 :  This  dataset  was  obtained  from the HybriDetector repository (accessed on 16 December 2024). It includes a class-imbalanced (1:10) training set (https://github.com/ML-Bioinfo-CEITEC/HybriDetector/ blob/main/ML/Datasets/miRNA\_train\_set.tsv) and a classbalanced  (1:1)  test  set  (https://github.com/ML-BioinfoCEITEC/HybriDetector/blob/main/ML/Datasets/miRNA\_ test\_set\_1.tsv).
- b) miRAW : This dataset was retrieved from the repository associated  with  (Pla et  al. 2018)  (https://bitbucket.org/ bipous/miraw\_data/src/master/ (accessed on 16 December 2024)), using the class-balanced (1:1) training set (miraw\_ data/PLOSComb/Data/ValidTargetSites/allTrainingSites. txt) and test set (miraw\_data/PLOSComb/Data/TestData/ balanced10/randomLeveragedTestSplit\_0.csv).
- c) Yang et al dataset : Data were downloaded from the repository  associated  with  the  article  (http://cosbi2.ee. ncku.edu.tw/mirna\_binding/download).

For each dataset, we extracted the mature miRNA sequences  from  each  example.  We  then  computed k -mer  counts ( k ¼ 3) from these sequences, generating a feature vector for each miRNA. These k -mer counts served as the sole input features for our classification models, allowing us to isolate the effect of miRNA frequencies. We trained a decision tree classifier on the training set of each dataset, using the miRNA k -mer counts as features and the original class labels (positive or negative) as targets. The trained models were then evaluated on their corresponding test sets. As a baseline, we also implemented a random classifier that assigns uniformly distributed values between 0 and 1 to each example, representing random guessing.

We evaluated model performance using the average precision score (APS) instead of the area under the precision-recall curve (AUPRC). While AUPRC is a common metric, it can be susceptible to overestimating performance when dealing with skewed score distributions or models that produce a limited range of prediction scores. APS, in contrast, provides a more robust measure of performance in such cases. It is calculated as the weighted mean of precision values at each prediction threshold, where the weights are the differences in recall between consecutive thresholds. Formally, APS is given by:

<!-- formula-not-decoded -->

where Pn and Rn are  the  precision  and  recall  at  the n th threshold.

## 2.2 Curating datasets corrected for miRNA frequency class bias

Publicly available data from high depth profiling of miRNAtargets  with  the  new  chimeric  eCLIP  experimental  method described by Manakov et  al. (2022) was downloaded from the  Gene  Expression  Omnibus  under  SubSeries  GEO  ID GSE198250  (on  14  May  2024).  The  samples  selected  for download were to include only chimeric eCLIP data from the human cell line HEK293T. Only files for the R1 read sequences were used, yielding one file per sample to be processed. The files were selected using the GEOparse Python package (version  2.0.4)  and  downloaded  using  enaBrowserTools available on GitHub (https://github.com/enasequence/ enaBrowserTools (accessed May 2024)).

The downloaded samples were pre-processed as per part of the pipeline for the analysis of total chimeric eCLIP datasets described by Manakov et al. (2022). The utilized part of the pipeline, available on their GitHub, trims the 3 0 adapters, and trims  10  nucleotides  from  the  3 0 end to  ensure  no  random sequences from the 3 0 UMI remain.

The  sample  files  were  then  processed  by  HybriDetector (Hejret et  al. 2023),  a  chimeric  read  annotation  pipeline available on GitHub (https://github.com/ML-BioinfoCEITEC/HybriDetector), to filter and separate different types of reads. The pipeline outputs high confidence pairs of guide (non-coding RNA) to target (binding site) sequences. To handle larger files from Manakov et al. (2022), some modifications were made to the original pipeline, and have been made available on GitHub (https://github.com/ML-BioinfoCEITEC/HybriDetector/tree/fix\_clustering).  A  total  of  19 sample files were successfully processed with HybriDetector. The  files  were  vertically  concatenated  into  a  single  dataset containing  all  the  examples  of  the  positive  class  and  made publicly  available  on  Zenodo  (https://zenodo.org/records/ 14501607/files/AGO2\_eCLIP\_Manakov2022\_full\_dataset. tsv.gz).

Two  corresponding, publicly available datasets, from Klimentov � a et al. (2022) and Hejret et al. (2023), also processed using the HybriDetector pipeline, and each containing the  positive  class  curated  from  data  derived  from  chimeric eCLIP and CLASH  experimental methods, respectively, were downloaded from GitHub (https://github.com/ML -Bioinfo-CEITEC/miRBind/blob/main/Datasets/AGO2\_eCLIP\_ Klimentova22\_full\_dataset.tsv) and (https://github.com/MLBioinfo-CEITEC/HybriDetector/blob/main/ML/Datasets/AGO 2\_CLASH\_Hejret2023\_full\_dataset.tsv), respectively, producing  a  total  of  three  positive  class  datasets  (Klimentova2022, Hejret2023, and Manakov2022), for each of which the negative class was generated in silico .

These three positive class datasets were processed via a series of post-processing pipelines available on GitHub (https:// github.com/BioGeMT/miRBench\_paper/tree/v0.2.0/code/post\_

miRBench i545

process), primarily to generate the negative class and create the train/test splits.

Specifically, the datasets were first filtered to retain only sequence  pairs  for  which  the  non-coding  RNA  is  a  miRNA, eliminating other annotated non-coding RNA types, such as tRNAs  and  yRNAs.  Next,  the  datasets  were  deduplicated based  on  the  combination  of  miRNA  to  binding  site  sequence, to ensure unique positive examples. Positive examples  with  miRNA  families  unique  to  the  Manakov2022 dataset were extracted and labeled as the Manakov2022 leftout dataset. This separate dataset is useful for evaluating the generalizability of models trained on any of the train sets presented here.

Next, a series of steps were performed to generate the negative examples. For each of the datasets, binding site sequences with &gt; 90% similarity were clustered together. The majority of clusters were singletons (Manakov2022 60%, Hejret2023 85%, Klimentova2022 84%), with the largest cluster found in  the  Manakov2022  dataset  consisting  of  253  sequences (0.02% of sequences). Then, to prevent the bias identified in previous datasets, negatives were generated per miRNA family in a way that retains the same proportion of miRNA families in each of the positive and negative classes. The dataset was alphabetically  sorted  by  the  miRNA  family  name.  For each group of positive examples with miRNAs in the same miRNA family, negative examples with binding sites were selected. The selected target sequences were selected only from target clusters not overlapping with any target cluster of the positive  miRNA  family.  These  candidates  were  randomly sampled, allowing only one binding site per cluster to be included  in  the  negative  examples  for  that  miRNA  family group. The described procedure ensures that miRNA binding site pairs in the negative class are sufficiently different from those in the positive class. A positive to negative class ratio of 1:1 was produced for each dataset.

The Hejret and Manakov datasets were then split into train and test sets according to which chromosome the binding site sequence maps to. Binding site sequences from chromosome 1 were assigned to the test set and the rest were assigned to the train set, to ensure binding sites in the train and test sets were sufficiently dissimilar.

This series of post-processing pipelines generated a total of six  comprehensive  datasets; Klimentova2022 test  set  (477 positives), Hejret2023 train  (4084  positives)  and  test  (495 positives) sets, and Manakov2022 train (1 253 320 positives), test  (168 342  positives),  and  left-out  (10 027  positives)  sets consisting of miRNAs not found in any of the other datasets.

## 2.3 State-of-the-art: miRNA binding site prediction models

TargetScanCnn\_McGeary2019 (McGeary et al. 2019) is a  convolutional  neural  network  with  two  convolutional layers and two fully connected layers. It uses an outer product  of  a  one-hot-encoded  representation  of  the  first  10 nucleotides  of  a  miRNA  and  12  nucleotides  of  its  putative target. All possible 12 nucleotide subsequences of the putative target are scored and the highest score is used as a final prediction. TargetScanCnn was trained using RNA Bindn-Seq (RBNS)-derived dissociation constants ( K d) and mRNA-transfection fold-change measurements produced by the  authors.  The  model  predicts -log( K d)  between  miRNA and putative target, which can be used as a proxy for a binding affinity score.

CnnMirTarget\_Zheng2022 (Zheng et  al. 2020) uses a convolutional  neural  network  consisting  of  four  convolutional layers with max-pooling followed by two dense layers with  dropout.  It  uses  a  one-hot-encoded  representation  of miRNA concatenated with a putative target, padded to 110 nucleotides. CnnMirTarget was trained on data from three sources:  (1)  human  AGO1  CLASH  Helwak  2013  data (Helwak et al. 2013); (2) Caenorhabditis elegans and mammalian  iPAR-CLIP  data  (Grosswendt et  al. 2014);  and  (3) MirTarBase  data  with  strong  experimental  evidence  (Hsu et al. 2011). The model's predictions may be directly used to measure the probability of binding.

TargetNet\_Min2021 (Min et al. 2022) is a deep learning model based on the ResNet architecture (He et al. 2015). It uses  a  one-hot-encoded representation of  a miRNA and its putative  target  after  extended  seed  alignment. TargetNet was trained  on  the  miRAW  dataset  (Pla et  al. 2018).  This dataset was constructed from four sources: (1) Diana TarBase (Vlachos et al. 2015); (2) MirTarBase (Chou et al. 2016);  (3)  human  AGO1  CLASH  Helwak  2013  data (Helwak et  al. 2013);  and  (4) C.  elegans iPAR-CLIP  data (Grosswendt et al. 2014). The model's predictions may be directly used to measure the probability of binding.

miRBind\_Klimentova2022 (Klimentov � a et al. 2022) is a deep learning model based on the ResNet architecture (He et  al. 2015).  It  uses  a  two-dimensional  representation  of miRNA and putative target sequence. To address the imbalance between positive and negative classes, miRBind uses an instance  hardness-based  label  smoothing  approach,  which keeps important, discriminative examples in the final training dataset and discards easily classifiable examples to rebalance the  skewed  ratio  (Guo  and  Viktor  2004). miRBind was trained on AGO1 CLASH Helwak 2013 data (Helwak et al. 2013). The model's predictions are directly used to measure the probability of binding. miRNA\_CNN\_Hejret2023 (Hejret et al. 2023) is a convolutional neural network consisting  of  six  convolutional  layers  followed  by  dense  layers.  It uses a two-dimensional representation of miRNA and putative binding site interaction introduced in miRBind (Klimentov � a et  al. 2022). miRNA\_CNN\_Hejret2023 was trained on the AGO2 CLASH Hejret 2023 training set that is also presented in this article. The model's predictions are directly used to measure the probability of binding.

InteractionAwareModel\_Yang2024 (Yang et al. 2024) is  a  deep  multi-head  attention  network  consisting  of three  parts:  (1)  sequence  feature  extraction;  (2)  interaction pattern identification; and (3) classification. It uses a one-hotencoded  representation  of  miRNA  sequence  padded  to  30 nucleotides and the first 40 nucleotides of a putative target. The InteractionAwareModel was trained on re-analyzed data  from  human  AGO1  CLASH  Helwak  2013  (Helwak et  al. 2013). This model's predictions may also be used directly to measure the probability of binding.

Beyond  these  ML  methods  that  have  been  specifically trained for the task, we also implemented two simple methods based on co-folding and seed. RNACofold is a tool offered by the ViennaRNA Package (Lorenz et al. 2011). It computes the binding  energy  and  base-pairing  pattern  of  an  input  pair  of interacting RNA molecule sequences. The model's output predicted minimum free energy of the folded structure is multiplied by -1 to be used as a measure of binding affinity.

For the seed measure, we followed previous studies (McGeary et  al. 2022) and four different  definitions  of  seed

i546 Sammut et al.

were used: (1) Seed8mer -full complementarity on positions 2-8 and A on position 1; (2) Seed7mer -full complementarity on positions 2-8, or full complementarity on positions 2-7 and A on the position 1; (3) Seed6mer -full complementarity on positions 2-7, or full complementarity on positions 3-8 or full complementarity on positions 2-6 and A on the position 1; and (4) SeedNonCanonical -the same as Seed6mer ,  but allowing for a single bulge or mismatch.

## 2.4 Retraining the miRNA\_CNN\_Hejret2023 model

We retrained the miRNA\_CNN\_Hejret2023 model on some of the new bias-corrected datasets presented here. To evaluate the effect of correcting the miRNA frequency class bias on the performance  of  the miRNA\_CNN\_Hejret2023 model,  we retrained the model on the corrected Hejret2023 train set, to compare its performance with that of the original model that was trained on the (biased) Original\_Hejret2023 dataset.

Next, we investigated the reliance of the performance of the model on dataset size. We retrained the miRNA\_CNN\_ Hejret2023 model on five datasets of increasing sizes,  subsampled  from  the  Manakov2022  train  set,  and  on  the  full Manakov2022 train set. The subsampling was done randomly while keeping the positive: negative ratio exactly 1:1. The number  of  positives  in  the  training  datasets  are:  100,  1000,  4084 (equivalent  to  the  number  of  positives  in  the  bias-corrected Hejret2023  train  set  presented  here),  10 000,  100 000,  and 1253320 (the full Manakov2022 train set). Retraining of the miRNA\_CNN\_Hejret2023 model for all dataset sizes was carried out in the same way as in the original article including all the hyperparameter settings (Hejret et al. 2023).

## 3 Results and discussion

## 3.1 microRNA frequency class bias can affect target site classification performance

A critical issue hindering accurate miRNA binding site prediction is the microRNA frequency class bias . This bias arises when certain miRNAs appear disproportionately more often in  the  positive  class  than  in  the  negative  class  of  published datasets. This skewed representation is often exacerbated by the  common  practice  of  generating  negative  examples  by pairing  random  miRNAs  with  binding  sites.  Consequently, highly abundant miRNAs in the positive class are often underrepresented in the negative set.

To  investigate  the  impact  of  this  issue,  we  conducted  a sanity test using a previously published (Hejret et al. 2023) chimeric  read  dataset  (Original\_Hejret2023).  We  used  the train  set  to  train  a  simple  decision  tree  machine  learning model using only the miRNA sequence as input, excluding any binding site information and effectively isolating the influence of miRNA frequencies. This model cannot learn anything about miRNA binding rules as it is missing information about  the  miRNA binding site  interaction,  and  thus  everything the model learns is irrelevant to the biological function of  miRNA-target  interactions.  We  would  therefore  expect this model  to  perform  no  better  than  random  chance (Average Precision Score (APS) � 0.5). However, the reported APS  on  the  Original\_Hejret2023  test  set  was  much  higher (APS � 0.75). A similar analysis on another widely used dataset  (Pla et  al. 2018),  which  also  exhibits  this  miRNA  frequency class bias, produced the same effect, with the decision tree  model  trained  only  on  miRNA  sequences  significantly outperforming random chance (APS � 0.85). The same analysis performed in the datasets used to train the Interaction  Aware  Model  (Yang et al. 2024),  in  which miRNA  frequencies  were  control  for,  shows  no  predictive power for the simple model (APS � 0.43 j random APS ¼ 0.50).

These findings confirm that even simple models can exploit miRNA frequency  imbalances  between  classes  in  the  data, achieving  deceivingly  high  predictive  performance  on  the training data, but failing to generalize to new datasets. This miRNA frequency imbalance allows the models to learn spurious  patterns  in  the  miRNA  frequency  distribution  rather than the true underlying features of the data such as sequence complementarity, structural accessibility, or legitimate miRNA binding rules. This artifact poses a significant threat to model generalization. If a classifier relies on the disproportionate  presence  of  certain  miRNAs  in  the  positive  class, rather  than  actual  binding  patterns  in  the  miRNA  binding site interaction, it may fail to accurately predict miRNA binding  sites  when  tested  on  a  dataset  featuring  a  different miRNA frequency distribution.

## 3.2 Alternative negative example generation method can correct the miRNA frequency class bias

We have proceeded to correct the microRNA frequency class bias in one of the previously published datasets (Hejret et al. 2023) by implementing a novel procedure for negative example  generation  (Section  2.2).  Briefly,  instead  of  associating  a random miRNA to each positive binding site, we attach a carefully  selected  binding  site  to  each  instance  of  a  miRNA  that occurs in the positive class, effectively ensuring a perfect match in miRNA frequency between the positive and negative classes across the entire dataset. When we attempted to train another simple decision tree classifier, using only the miRNA sequence as input, on the new corrected Hejret2023 dataset, the performance (APS ¼ 0.39) was lower than that of a random model (APS ¼ 0.50). This demonstrates that our method of negative example generation successfully addresses the previously identified artifact. The slight underperformance, however, requires further explanation. It arises because, while the entire dataset maintains  a  perfect  balance  of  miRNA  frequencies  between positive  and  negative  examples,  the  train/test  split  strategy does not explicitly enforce this balance. In order to minimize data leakage, we use targets on chr1 as the test set, across all datasets. For example, the training set might, by chance, have slightly more positive examples for a particular miRNA than negative examples, then by default the test set will have more negative  than  positive  examples  for  the  same  miRNA  since their total number is equal. Since our simple decision tree can only learn from miRNA sequences and has no access to binding site information, it might learn this incidental imbalance in the  training  set  and  predict  'positive'  more  often  for  that miRNA. Therefore, the model's learned bias from the training set will lead it to perform worse than random on the test set, explaining the observed APS of 0.39. It is important to note that this underperformance is a consequence of the limitations of the simple decision tree model and the train/test split strategy and does not invalidate the effectiveness of our negative example  generation  method  in  correcting  the  miRNA  frequency class bias.

Having addressed the miRNA frequency bias, we considered the  possibility  of  inadvertently  introducing  a  similar  bias  related to the binding site sequences themselves. To investigate this,  we  trained  another  simple  decision  tree  classifier,  this

miRBench i547

time using only the binding site sequences as input and excluding any miRNA sequence information. This model achieved an APS of 0.50, comparable to the performance of a random classifier.  This  result  suggests  that  our  negative  example  generation  procedure  does  not  introduce  a  notable binding  site frequency  bias  between  the  positive  and  negative  classes. Therefore, we consider this procedure suitable for generating the negative class in miRNA: binding site datasets, promoting more robust training and benchmarking of models.

## 3.3 Machine learning prediction models are affected by microRNA frequency class bias

We  proceeded  to examine  whether  previously published miRNA binding site prediction models were affected by this artifact.  As  a  positive  control,  we  engineered  a  test  dataset based on the Hejret2023 test set, in which we only keep members  of  the  enriched  class  (positive  or  negative)  for  each miRNA.  This  dataset maximizes  the predicted effect of miRNA frequency class bias. Several models show an artificial improvement  in performance in this engineered dataset, evidencing that they have learned the miRNA class frequency bias (Table 1). We compared the APS of seven miRNA binding site  prediction tools on the uncorrected Original\_Hejret2023 dataset and the bias-corrected version (Hejret2023) presented here. This experiment demonstrates that several models are affected by the artifact, showing a notable drop in performance when evaluated in the corrected dataset (Table 1).

As expected, the model that was trained on the Original\_Hejret2023 dataset, miRNA\_CNN\_Hejret2023 , shows a large drop in performance when the artifact is corrected  (APS:  0.89 /uni21FE 0.77).  Our  'negative  control'  model, RNACofold ,  which  is  not  trained  on  any  miRNA  binding site interaction datasets, and therefore should not be affected by microRNA frequency class bias, shows a random variation  drop  in  APS  of  0.03  (APS:  0.77 /uni21FE 0.74) similar to  the InteractionAwareModel (APS:  0.77 /uni21FE 0.74) and the TargetScanCNN (APS:  0.74 /uni21FE 0.71),  while  a  completely random model shows a random increase of similar magnitude (APS:  0.49 /uni21FE 0.51).  Surprisingly,  we  also  notice  that  some other  models  show a  drop  in  performance  ( miRBind APS: 0.87 /uni21FE 0.80; TargetNet APS: 0.66 /uni21FE 0.58; CnnMirTarget APS:  0.63 /uni21FE 0.53).  This  is  surprising  because  these  models  were  trained  and  published  before  the Original\_Hejret2023 dataset was produced in 2023, and thus could not have been directly affected by the microRNA frequency class bias we have detected in this dataset. However, we notice that all of these models have been fully or partially

Table 1. Average precision score (APS) for several miRNA binding site prediction tools evaluated on the Engineered Hejret2023 dataset with inflated miRNA frequency class bias, the Original Hejret2023 test set, and on the bias-corrected Hejret2023 test set presented here.

| Tool                    |   (Engineered) Hejret2023 Majority class |   (Original) Hejret2023 test |   (Corrected) Hejret2023 test |
|-------------------------|------------------------------------------|------------------------------|-------------------------------|
| TargetScanCNN           |                                     0.7  |                         0.74 |                          0.71 |
| CnnMirTarget            |                                     0.73 |                         0.63 |                          0.53 |
| TargetNet               |                                     0.81 |                         0.66 |                          0.58 |
| miRBind                 |                                     0.98 |                         0.87 |                          0.8  |
| miRNA_CNN_Hejret        |                                     0.99 |                         0.89 |                          0.77 |
| Interaction Aware Model |                                     0.85 |                         0.77 |                          0.74 |
| RNACofold               |                                     0.86 |                         0.77 |                          0.74 |
| Random                  |                                     0.49 |                         0.49 |                          0.51 |

trained  on  an  older  AGO1  CLASH  dataset  (Helwak et  al. 2013), which originates from an experiment that also happens to be performed on the same cell line (HEK293) as the Hejret dataset, and thus contain similar microRNA frequency class bias as the newer Original\_Hejret2023 dataset. Therefore, the fact that models trained on other datasets also experienced  a  notable  drop  in  performance,  strongly  indicates  that  the  miRNA  frequency  class  bias  is  a  pervasive problem in the datasets currently available for miRNA binding site prediction model training. The widespread nature of this bias underscores the critical need for the development of corrected datasets, such as those presented in this study, to enable  the  training  of  accurate  and  generalizable  miRNA binding site prediction models.

## 3.4 Novel miRNA binding site datasets 3.4.1 Curation of datasets

Having  shown  that  the  majority  of  current  state-of-the-art methods are affected by artifacts produced from a family of related training/testing datasets, and recognizing the potential for other such datasets to introduce similar artifacts to future methods, we have produced a set of novel datasets with corrected  microRNA  frequency  class  bias.  We  curated  three miRNA binding site datasets; Hejret2023, Klimentova2022, and Manakov2022, produced by two different highthroughput  experimental  protocols  (CLASH,  and  chimeric eCLIP), and standardized through a post-processing pipeline (Section 2.2). The Hejret2023 and Klimentova2022 datasets are based on previously published datasets used in the training and/or evaluation of miRNA binding site prediction models  (Klimentov � a et al. 2022,  Hejret et al. 2023).  The Manakov2022 dataset is completely novel, orders of magnitude larger than the others combined, and, to our knowledge, has not yet been used by any model to date for training or evaluation. To further ensure that future models using these datasets generalize better, we initially extract a 'left-out' set from the Manakov2022 dataset. The 'left-out' set contains only miRNAs from miRNA families that are unique to the Manakov2022  dataset, and not found in any of the Hejret2023 and/or Klimentova2022 datasets. The 'left-out' dataset is a small dataset, but it is crucial to ensure the generalizability of models trained on the Hejret2023 or Manakov2022 train sets. The remaining Manakov2022 dataset  contains  miRNAs  from  miRNA  families  that  are  also found  in  the  Hejret2023  and/or  Klimentova2022  datasets. We  applied  the negative  example  generation  procedure (Section 2.2), that preserves the relative  abundance  of miRNAs in the positive and negative classes, to each of the Manakov2022 'left-out' set and the remaining Manakov2022 dataset,  and  further  split  the  latter  into  the 'train' and 'test' sets presented here.

## 3.4.2 Descriptive characteristics of positive samples in novel datasets

As  part  of  the  curation  process  of  the  three  collections  of datasets  described,  the  target  binding  site  sequences  were aligned to the human genome via HybriDetector (Hejret et al. 2023),  enabling  assignment  of  genic  feature  annotations (Fig. 1A). The Klimentova2022 shows only 23% of binding sites mapping to regions annotated as 3 0 UTR. The other two datasets (Hejret2023 and Manakov2022) show a consistent prevalence of 3 0 UTR binding sites at 37% and 41% of annotated  interactions,  respectively.  We  observe  that,  overall,

i548 Sammut et al.

chimeric  eCLIP  appears  to  have  more  intronic  targets  than CLASH, at least for our limited dataset size.

Interestingly, most miRNA-target prediction methods only take  into  consideration  targets  within  the  3 0 UTR,  rarely exons, and never introns, despite the binding site distribution seen here. This is a potential blind spot that ignores a large number, potentially a majority, of binding sites from consideration of their effect on the overall regulation of the target messenger RNA.

Given the predominance of seed-like measures for miRNA binding site prioritization, we explored the prevalence of canonical seed in binding site sequences in the produced datasets  (Fig.  1B).  Canonical  seed  interactions  comprised  a minority of each dataset: Manakov2022 (44%), Hejret2023 (30%), and Klimentova2022 (43%). For all three datasets, the Seed7mer was the most prevalent type of canonical seed

Figure 1. (A) Distribution of binding site percentage overlap to genomic element annotations. (B) Distribution of miRNA seed types.

<!-- image -->

interaction,  at  22%,  14%,  and  25%  of  interactions  in Manakov2022,  Hejret2023,  and  Klimentova2022,  respectively. Interestingly, interactions lacking exact match for any of the defined seeds comprised only approximately 22% of Manakov2022, and 28% for Hejret2023 and for Klimentova2022.  It  also  appears  that  the  chimeric  eCLIP methodology used in Manakov2022 and Klimentova2022 is more effective in  capturing canonical  seed-type  interactions compared to the CLASH method used in Hejret2023, at least for our limited dataset size.

## 3.5 Benchmarking and retraining models 3.5.1 Benchmarking of state-of-the-art miRNA binding site prediction tools on novel datasets

We evaluated multiple state-of-the-art binding site prediction methods including CNNs ( TargetScanCnn\_McGeary2019 , CnnMirTarget\_Zheng2022 , miRNA\_CNN\_Hejret2023 ), ResNets ( TargetNet\_Min2021 , miRBind\_Klimentova 2022 ), attention-based models ( InteractionAware Model\_Yang2024 ),  and  simpler  co-folding  and  seed-based strategies  ( RNACofold , Seed6mer , Seed7mer , Seed8mer , SeedNonCanonical ) on all datasets (Fig. 2, Table 2).

Within individual test sets, miRBind achieved the highest APS on the Hejret2023 and Klimentova2022 test sets (0.80; 0.75), while TargetScanCnn led on both Manakov test and left-out datasets (0.77, 0.76). When examining cross-dataset performance, TargetScanCnn demonstrated consistent performance across all datasets (0.71-0.77), miRBind showed strong  but  variable  performance  (0.71-0.80)  with  better results on Klimentova2022 and Hejret2023 tests, and miRNA\_CNN\_Hejret2023 maintained  stable  performance (0.71-0.77).  In  contrast, CnnMirTarget and TargetNet showed consistent but lower performance (0.51-0.58) across all datasets, while the InteractionAwareModel displayed moderate performance (0.63-0.74).

Overall, there is not a single state-of-the-art method that consistently outperforms all others in every dataset, leaving the field open for new methods to be developed.

## 3.5.2 Retrained miRNA\_CNN\_Hejret2023 benchmarked

To further confirm the importance of unbiased datasets, we retrained the miRNA\_CNN\_Hejret2023 model proposed by Hejret et al. (2023) exclusively on the new miRNA frequency class unbiased Hejret2023 train set. The intent of this exercise  is  to  understand  to  which  extent  the microRNA  frequency class bias in the original dataset affected the potential of the trained model  to  learn. We  elected  to use the

Figure 2. Precision-recall curves. All miRNA binding site prediction tools available on the miRBench package, evaluated on the Klimentova2022 test set, Hejret2023 test set, Manakov2022 test set, and Manakov2022 left-out set.

<!-- image -->

miRBench i549

miRNA\_CNN\_Hejret2023 model instead of the better performing miRBind\_Klimentova2022 architecture as it is a simpler model consisting of a single convolutional neural network. The miRBind model utilizes smaller pilot models and soft labeling of training data, and would need more optimization to be retrained effectively.

We retained unchanged architecture and hyperparameters from the original publication to isolate just the effect of training data quality. We cannot guarantee that this is the optimal model or optimal performance that this model could potentially learn from this dataset. That said, the retrained CNN achieved an APS of 0.86 on the Hejret2023 test set (versus original miRNA\_CNN\_Hejret2023 model APS ¼ 0.77, Table 3) showing significant predictive performance improvement  (Fig.  3).  This  retrained  model  outperformed  the  best performing state-of-the-art tools in all test datasets.

We noted that the retrained model performed much better on  the  Hejret2023  test  set  (APS ¼ 0.86)  compared  to  the other  testing  datasets  (APS ¼ 0.77-0.79).  This  leads  us  to believe  that  there  could  be  experiment-  or  dataset-specific

Table 2. Average precision score (APS) from evaluating seven miRNA binding site prediction tools against the four test datasets in miRBench.

| Tool                               |   Klimentova 2022 test |   Hejret 2023 test |   Manakov 2022 test |   Manakov 2022 left-out |
|------------------------------------|------------------------|--------------------|---------------------|-------------------------|
| TargetScanCNN                      |                   0.74 |               0.71 |                0.77 |                    0.76 |
| CnnMirTarget                       |                   0.52 |               0.53 |                0.53 |                    0.51 |
| TargetNet                          |                   0.53 |               0.58 |                0.57 |                    0.58 |
| miRBind                            |                   0.75 |               0.8  |                0.71 |                    0.71 |
| miRNACNN                           |                   0.74 |               0.77 |                0.71 |                    0.71 |
| Hejret2023 Interaction Aware Model |                   0.66 |               0.74 |                0.69 |                    0.63 |
| RNACofold                          |                   0.67 |               0.74 |                0.63 |                    0.65 |
| Random                             |                   0.51 |               0.51 |                0.52 |                    0.5  |

The best performing model per dataset is highlighted.

Table 3. Average precision score (APS) of original and retrained models.

| Tool                                |   Klimentova 2022 test |   Hejret 2023 test |   Manakov 2022 test |   Manakov 2022 left-out |
|-------------------------------------|------------------------|--------------------|---------------------|-------------------------|
| Original miRNA CNN_Hejret2023       |                   0.74 |               0.77 |                0.71 |                    0.71 |
| Retrained on Hejret 2023 train set  |                   0.77 |               0.86 |                0.79 |                    0.78 |
| Retrained on Manakov 2022 train set |                   0.84 |               0.84 |                0.84 |                    0.81 |

The best in each dataset is highlighted.

0

elements learnt by the model that still harm full generalizability.  The Hejret2023 dataset is the only one produced using the CLASH experimental technique, while all other datasets are produced using the chimeric eCLIP technique. Moreover, the  Hejret2023  dataset  appears  to  have  retrieved  a  higher percentage of No Seed ('None') and 'Non-Canonical Seed' interactions (Fig.  1B).  Thus,  models  trained  on  this  dataset may overestimate  the  importance  of  these  types  of  interactions, harming generalizability to other datasets.

Given that some experiment- or dataset-specific elements remain elusive, we advocate for reporting results on all datasets independently in future benchmarking efforts. We expect that incorporating additional experimental datasets into future  iterations  of  miRBench  will  help  uncover  experimentand  dataset-specific  biases  underlying  these  performance discrepancies.

In addition to the retraining on the Hejret2023 dataset, we also retrained the miRNA\_CNN\_Hejret2023 model on the larger,  unbiased  Manakov2022  train  set,  achieving  strong performance across all  four  test  datasets  (APS ¼ 0.81-0.84, Fig. 3, Table 3). This demonstrates that even with a simple CNN architecture, a well-curated, large training set can yield state-of-the-art results.

## 3.5.3 Increasing dataset size improves CNN performance with diminishing results

The Manakov2022 train dataset, with 1253320 positive examples, is approximately 325 times larger than the Hejret2023 train set (4084 positive examples). To investigate the influence of training dataset size on model performance, we conducted a series of experiments by retraining the miRNA\_CNN\_Hejret2023 model on subsets of the Manakov2022 train set, ranging from 100 to 1253320 positive examples.

As illustrated in Fig. 4, model performance, measured by APS, increased logarithmically with the number of training examples. However, this improvement plateaued once the training set size reached  approximately  4084  positive  examples,  the  size  of  the corrected Hejret2023 train set. Further increasing the dataset size beyond this point yielded only marginal gains, with performance stabilizing at an APS of approximately 0.84 for the three test sets and 0.81 for the Manakov2022 left-out dataset. These findings suggest that while larger datasets are generally beneficial, there are diminishing returns beyond a certain threshold, at least for this specific model architecture.

While our results demonstrate the effectiveness of a relatively simple CNN architecture, it is conceivable that larger, more  complex  models  could  extract  further  insights  and achieve even higher performance from these extensive datasets. However, exploring the interplay between model

Figure 3. Precision-recall curves. Retrained models (CNN retrained Hejret2023 and CNN retrained Manakov2022), as well as models that previously performed best on any benchmarked dataset (miRBind, TargetScanCnn), evaluated on the Klimentova2022 test set, Hejret2023 test set, Manakov2022 test set, and Manakov2022 left-out set.

<!-- image -->

i550 Sammut et al.

Figure 4. Average precision score against training dataset size for retrained model.

<!-- image -->

complexity and very large dataset sizes is beyond the scope of this current work, and we encourage further investigation in this area by other research groups.

## 4 Conclusion

This study introduces miRBench, a novel benchmark framework for evaluating miRNA binding site prediction methods, and  identifies  a  critical  and  previously  unreported  bias  in existing  datasets:  miRNA  frequency  class  bias.  We  demonstrate  that  this  bias,  stemming  from  the  conventional  approach to generating negative training examples, artificially inflates the performance of several state-of-the-art prediction models. To address this issue, we developed a new methodology for negative example generation that effectively mitigates this bias, ensuring a more balanced representation of miRNA families in both positive and negative classes.

Using this  improved  methodology we curated several new datasets, including the Manakov2022 dataset which is derived from a novel experimental technique and is orders of magnitude larger than any previously available miRNA binding site interaction  dataset.  Our  comprehensive  benchmarking  on these datasets revealed that while several current methods perform  reasonably  well,  no  single  method  consistently  outperforms the others across all datasets. Importantly, we show that a simple convolutional neural network model, when retrained on  our  unbiased  datasets,  surpasses  the  performance  of  all existing methods on these benchmarks, highlighting the profound impact of dataset quality on model performance.

Furthermore, our investigation  into  the  relationship  between dataset size and model performance revealed that while increasing the number of training examples generally improves performance, this improvement plateaus beyond a certain threshold, at least for the CNN architecture employed in this study. This suggests that dataset quality, particularly the absence of biases, is as crucial as, if not more important than, dataset size. The consistently lower performance of models on the Manakov2022 'leftout'  dataset,  which  contains  unique  miRNA  families,  underscores the importance of diverse and representative training data for achieving true model generalizability.

To  facilitate  future  research,  we  provide  miRBench,  an open-source Python package that offers easy access to the curated datasets, encoders, and implementations of state-of-theart prediction models. We believe that miRBench, with its unbiased  datasets  and  user-friendly  interface,  will  serve  as  a valuable  resource  for  the  development  and  evaluation  of more  accurate  and  robust  miRNA  binding  site  prediction methods. Future work should focus on exploring more complex model architectures capable of leveraging the full potential  of  large-scale  datasets,  as  well  as  further  investigation into the dataset- and experiment-specific factors that contribute to the remaining performance discrepancies. Ultimately, a deeper understanding of miRNA-target interactions, driven by rigorous benchmarking and unbiased datasets, will be crucial for unlocking the full potential of miRNAs in both basic biology and clinical applications.

## Acknowledgements

The authors would like to thank E. Maragkakis, M. Ciach, A. Balestrucci, and V. Mart � ınek for their constructive criticism of the article.

## Author Contributions

Stephanie Sammut: Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing - original draft, Writing - review &amp; editing, Katarina Gresova: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing - original draft, Writing review &amp; editing, Dimosthenis Tzimotoudis: Data curation, Formal analysis, Investigation, Methodology, Software, Validation, Visualization, Writing - original draft, Writing review  &amp;  editing,  Eva  Marsalkova:  Data  curation,  Formal analysis,  Investigation,  Methodology,  Software,  Validation, Visualization, Writing - original draft, Writing - review &amp; editing,  David  Cechak:  Data  curation,  Formal  analysis, Investigation, Methodology, Software, Validation, Visualization, Writing - original draft, Writing - review &amp; editing,  Panagiotis Alexiou: Conceptualization, Funding acquisition,  Investigation,  Methodology,  Project  administration, Resources, Supervision, Visualisation, Writing -original draft, Writing - review &amp; editing

Conflict of interest: none declared.

## Funding

This  work  was  supported  by  funding  from  the  projects BioGeMT (HORIZON-WIDERA-2022 Grant ID: 101086768) at the University of Malta and miRBench RNS-2024-022 for 'Collaboration  for  microRNA  Benchmarking'  from  Xjenza Malta awarded to Panagiotis Alexiou. Scientific data presented in this article was obtained with the help of the Bioinformatics Core Facility of CEITEC Masaryk University supported by the NCMG  Research  Infrastructure (LM2023067  funded by MEYS  CR).  This  work  was  supported  by  computational resources  provided  by  the  TargetID  project,  Novel  Drug Targets for Infectious Diseases, funded by the Malta Council for  Science  and  Technology  (MCST)  COVID-19 R&amp;D Fund 2020 (Grant COV.RD.2020-11), and the e-INFRA CZ project (ID:  90254),  supported  by  the  Ministry  of  Education,  Youth and Sports of the Czech Republic.

## Data availability

The latest version of the miRBench package is freely available on  GitHub  at  https://github.com/katarinagresova/miRBench/

miRBench i551

tree/v1.0.1, and has a permanent version deposited on Zenodo at https://zenodo.org/records/15084661. The Python package is also  distributed  through  the  Python  Package  Index  (PyPI)  at https://pypi.org/project/miRBench/1.0.1/. The datasets produced were deposited on Zenodo at https://zenodo.org/records/ 14501607,  so  they  are  easily  accessible  by  the  miRBench Python  package.  Code  for  processing  datasets  is  available  at https://github.com/BioGeMT/miRBench\_paper/tree/v1.0.1.

## References

- Alexiou P, Maragkakis M, Papadopoulos GL et al. Lost in translation: an assessment and perspective for computational microRNA target identification. Bioinformatics 2009; 25 :3049-55.
- Bartel DP. MicroRNAs: genomics, biogenesis, mechanism, and function. Cell 2004; 116 :281-97.
- Bernstein E, Kim SY, Carmell MA et al. Dicer is essential for mouse development. Nat Genet 2003; 35 :215-7.
- Broughton  JP,  Lovci  MT,  Huang  JL et  al. Pairing  beyond  the  seed supports microRNA targeting specificity. Mol Cell 2016; 64 :320-33.
- Calin GA, Croce CM. MicroRNA signatures in human cancers. Nat Rev Cancer 2006; 6 :857-66.
- Chakraborty C, Sharma AR, Sharma G et al. Therapeutic miRNA and siRNA: moving from bench to clinic as next generation medicine. Mol Ther Nucleic Acids 2017; 8 :132-43.
- Chou C-H, Chang N-W, Shrestha S et al. miRTarBase 2016: updates to the experimentally validated miRNA-target interactions database. Nucleic Acids Res 2016; 44 :D239-47.
- Condrat CE, Thompson DC, Barbu MG et al. miRNAs as biomarkers in disease: Latest findings regarding their role in diagnosis and prognosis. Cells 2020; 9 :276. https://doi.org/10.3390/cells9020276
- Dai R, Ahmed SA. MicroRNA, a new paradigm for understanding immunoregulation,  inflammation,  and  autoimmune  diseases. Transl Res 2011; 157 :163-79.
- Didiano D, Hobert O. Perfect seed pairing is not a generally reliable predictor for miRNA-target interactions. Nat Struct Mol Biol 2006; 13 :849-51.
- Esquela-Kerscher A, Slack FJ. Oncomirs-microRNAs with a role in cancer. Nat Rev Cancer 2006; 6 :259-69.
- Gre � sov � a K, Alexiou P, Giassa I-C et al. Small RNA targets: advances in prediction  tools  and  high-throughput  profiling. Biology  (Basel) 2022; 11 :1798.
- Grosswendt S, Filipchyk A, Manzano M et al. Unambiguous identification of miRNA: target site interactions by different types of ligation reactions. Mol Cell 2014; 54 :1042-54.
- Guo H, Viktor HL. Learning from imbalanced data sets with boosting and data generation. SIGKDD Explor 2004; 6 :30-9.
- H � ebert SS, De Strooper B. Alterations of the microRNA network cause neurodegenerative disease. Trends Neurosci 2009; 32 :199-206.
- Hejret V, Varadarajan NM, Klimentova E et al. Analysis of chimeric reads characterises the diverse targetome of AGO2-mediated regulation. Sci Rep 2023; 13 :22895.
- He K et al. Deep residual learning for image recognition. arXiv [cs.CV]. https://doi.org/10.48550/arXiv.1512.03385, 2015,  preprint: not peer reviewed.
- He L, Hannon GJ. MicroRNAs: small RNAs with a big role in gene regulation. Nat Rev Genet 2004; 5 :522-31.
- Helwak A, Kudla G, Dudnakova T et al. Mapping the human miRNA interactome by CLASH reveals frequent noncanonical binding. Cell 2013; 153 :654-65.
- Hsu S-D, Lin F-M, Wu W-Y et al. miRTarBase: a database curates experimentally validated microRNA-target interactions. Nucleic Acids Res 2011; 39 :D163-9.
- Ikeda S, Kong SW, Lu J et al. Altered microRNA expression in human heart disease. Physiol Genomics 2007; 31 :367-73.
- Ivey KN, Srivastava D. MicroRNAs as regulators of differentiation and cell fate decisions. Cell Stem Cell 2010; 7 :36-41.
- Klimentov � a E, Hejret V, Kr � cm � a � r J et al. miRBind: a deep learning method for miRNA binding classification. Genes (Basel) 2022; 13 :2323.
- Lee RC, Feinbaum RL, Ambros V et al. The C. elegans heterochronic gene lin-4 encodes small RNAs with antisense complementarity to lin-14. Cell 1993; 75 :843-54.
- Liu J, Carmell MA, Rivas FV et al. Argonaute2 is the catalytic engine of mammalian RNAi. Science 2004; 305 :1437-41.
- Lorenz R, Bernhart SH, H € oner Zu Siederdissen C et  al. ViennaRNA package 2.0. Algorithms Mol Biol 2011; 6 :26.
- Manakov SA et al. Scalable and deep profiling of mRNA targets for individual microRNAs with chimeric eCLIP. bioRxiv, https://doi.org/ 10.1101/2022.02.13.480296, 2022, preprint: not peer reviewed.
- McGeary SE, Bisaria N, Pham TM et al. MicroRNA 3 0 -compensatory pairing occurs through two binding modes, with affinity shaped by nucleotide identity and position. Elife 2022; 11 :e69803.
- McGeary SE, Lin KS, Shi CY et al. The biochemical basis of microRNA targeting efficacy. Science 2019; 366 :eaav1741.
- Min  S,  Lee  B,  Yoon  S et  al. TargetNet:  functional  microRNA  target prediction with deep neural networks. Bioinformatics 2022; 38 :671-7.
- Morita S, Horii T, Kimura M et  al. One Argonaute family member, Eif2c2 (Ago2), is essential for development and appears not to be involved in DNA methylation. Genomics 2007; 89 :687-96.
- O'Connell RM, Taganov KD, Boldin MP et al. MicroRNA-155 is induced  during  the  macrophage  inflammatory  response. Proc  Natl Acad Sci U S A 2007; 104 :1604-9.
- Pla A, Zhong X, Rayner S et  al. miRAW: a deep learning-based approach to predict microRNA targets by analyzing whole microRNA transcripts. PLoS Comput Biol 2018; 14 :e1006185.
- Rad SMAH, Halpin JC, Tawinwung S et al. MicroRNA-mediated metabolic reprogramming of chimeric antigen receptor T cells. Immunol Cell Biol 2022; 100 :424-39.
- Van Nostrand E, Pratt G,  Shishkin A et al. Robust transcriptome wide discovery  of  RNA-binding  protein  binding  sites  with  enhanced CLIP  (eCLIP). Nat  Methods 2016; 13 :508-14.  https://doi.org/10. 1038/nmeth.3810
- van Rooij E, Sutherland LB, Liu N et al. A signature pattern of stressresponsive  microRNAs  that  can  evoke  cardiac  hypertrophy  and heart failure. Proc Natl Acad Sci U S A 2006; 103 :18255-60.
- van Rooij E, Olson EN. MicroRNA therapeutics for cardiovascular disease:  opportunities  and  obstacles. Nat  Rev  Drug  Discov 2012; 11 :860-72.
- Rupaimoole R, Slack FJ. MicroRNA therapeutics: towards a new era for  the  management of cancer and other diseases. Nat Rev Drug Discov 2017; 16 :203-22.
- Shen L, Yang J, Zuo C et al. Circular mRNA-based TCR-T offers a safe and effective therapeutic strategy for treatment of cytomegalovirus infection. Mol Ther 2024; 32 :168-84.
- Sonkoly E, Pivarcsi A. Advances in microRNAs: implications for immunity and inflammatory diseases. J Cell Mol Med 2009; 13 :24-38.
- Thum T, Condorelli G. Long noncoding RNAs and microRNAs in cardiovascular pathophysiology. Circ Res 2015; 116 :751-62.
- Vlachos  IS,  Paraskevopoulou  MD,  Karagkouni  D et al. DIANATarBase v7.0: indexing more than half a million experimentally supported  miRNA:  mRNA  interactions. Nucleic  Acids  Res 2015; 43 :D153-9.
- Yang T-H, Chen J-C, Lee Y-H et al. Identifying human miRNA target sites  via  learning  the  interaction  patterns  between  miRNA  and mRNA segments. J Chem Inf Model 2024; 64 :2445-53.
- Zhao Y, Samal E, Srivastava D et al. Serum response factor regulates a muscle-specific microRNA that targets Hand2 during cardiogenesis. Nature 2005; 436 :214-20.
- Zheng X, Chen L, Li X et al. Prediction of miRNA targets by learning from interaction sequences. PLoS One 2020; 15 :e0232578.

© The Author(s) 2025. Published by Oxford University Press.

This is an Open Access article distributed under the terms of the Creative Commons Attribution License (https://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse, distribution, and reproduction in any medium, provided the original work is properly cited.

Bioinformatics, 2025, 41, 542-551

https://doi.org/10.1093/bioinformatics/btaf233

ISMB/ECCB 2025 Supplement
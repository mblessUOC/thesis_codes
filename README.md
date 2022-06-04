# Explanation
In this repositories you will find the codes created for the realization of the thesis *Deep learning model for tumor type classification using multiple omics* which was done on 2022.

You will find 2 brach folders: **RStudio Codes** and **Colab Codes**.

First folder are the necessary codes to download and preprocess 33 tumour types from Genomic Data Commons Data Portal (https://portal.gdc.cancer.gov/) from project The Cancer Genome Atlas Program
(https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga). Please follow the numeric file name which is intended to guide through the process.

Second folder you will find the Jupyter notebooks excetuded in Colab (https://colab.research.google.com/?hl=es) to exceute deep learning algorythm coded in Keras. You will find 5 files. Files that begins with *VAE* were designed to extract intrinsic data from those samples that doesnt have their molecular profile done through the 3 omics. The weight of this 3 models were used for transfer learning. The other 2 are the overall algorithms that are responsible to the analysis of the samples across the 3 omics.

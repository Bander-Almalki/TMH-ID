# TMH-ID: Predicting Interface Residues in Alpha-Helical Transmembrane Proteins Homodimers Using Sequential and Structural Features

<img src="https://github.com/Bander-Almalki/TMH-ID/main/output/Interface_Residues.png" alt="Interface_Residues" width="398" height="349">

A machine learning model that integrates two distinct computational approaches by combining sequential and structural features extracted from a large protein language model and a molecular dynamics simulation model, respectively. Additionally, it incorporates specialized transmembrane protein features to enhance the accurate identification of interface residues in transmembrane proteins.

## Installation

Require python>=3.7

```
git clone https://github.com/Bander-Almalki/TMH-ID.git
cd TMH-ID
conda env create -f environment.yml
conda activate TMH-ID
```

## Usage

- The Data folder contains the dataset in fasta files. This folder is divided into 3 subfolders representing the extraction method of the dimer (Crystal, NMR or ETRA).
- The main scripts can be found in the scripts folder and can be divided into two groups:
    - The first group is used to compare 3 different protein structure prediction models, on the NMR and Crystal datasets, to predict the interface residue. Two of these models are machine learning-based models (AlphaFold2Multimer and RossettaFolde2) and the other is a molecular dynamic simulation-based model (PREDDIMER).
    - The second group is used to compare our model *TMH-ID* with 3 other models, on the whole dataset, in predicting the interface residues in alpha-helical TM proteins homodimers.Two of these models are large protein language models (ProtBert and MSA Transformer) and the other represents the current state-of-the-art in the field.

The following commands are executed from the main repository **TMH-ID** to measure the accuracy of the models.

- To measure the accuracy of the structure-based models in predicting the interface residues, run the following:

```
python scripts/measuring_alphafold2mulitmer_dimers_prediction_accuracy.py
python scripts/measuring_rossettafold2_dimers_prediction_accuracy.py
python scripts/measuring_preddimer_dimers_prediction_accuracy.py
```

- To measure the accuracy of the TMH-ID model and the other models, run the following:

```
python scripts/measuring_TMH_ID_predictio_performane_on_interface_prediction.py
python scripts/measuring_msat_score_predictio_performane_on_interface_prediction.py
python scripts/measuring_protbert_embedding_prediction_performance_on_dimers_interface.py
```

- All the features used by the model can be found in the *Features* folder.
- TMH-ID weights can be found in the *TMH-ID_Model_Weights* Folder.

* * *
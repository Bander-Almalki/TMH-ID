# -*- coding: utf-8 -*-
"""Measuring_ProtBert_embedding_prediction_performance_on_dimers_interface



# Measuring_ProtBert_embedding_prediction_performance_on_dimers_interface
- The goal is to measure how good is protbert embedding in predicting interface residues in dimers
- the input is the prtobert embedding of a dimer(only the embedding of residues of a single helix).
- the output is whether the residue is an interface residue(1) or not (0)
"""


import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
"""## importing prtobert embedding of dimers' single helix"""


with open ('ProtBert_Sequence_Embedding/crystal/crys_prot_ProtBert_embaddings.pkl', 'rb') as f:
  crystal_protbert = pickle.load(f)

with open ('ProtBert_Sequence_Embedding/NMR/NMR_prot_ProtBert_embaddings.pkl', 'rb') as f:
  NMR_protbert = pickle.load(f)

with open ('ProtBert_Sequence_Embedding/ETRA/ETRA_prot_ProtBert_embaddings.pkl', 'rb') as f:
  ETRA_protbert = pickle.load(f)

crystal_protbert.keys()

NMR_protbert.keys()

ETRA_protbert.keys()

#all protbert sequences
all_protbert = crystal_protbert | NMR_protbert | ETRA_protbert

all_protbert.keys()

"""## Importing interface labels"""


with open ('Features/Dimers_interface_labels/crys_proteins_interface_labels.pkl', 'rb') as f:
  crys_label = pickle.load(f)
with open ('Features/Dimers_interface_labels/NMR_proteins_interface_labels.pkl', 'rb') as f:
  NMR_lable = pickle.load(f)
with open ('Features/Dimers_interface_labels/ETRA_proteins_interface_labels.pkl', 'rb') as f:
  ETR_label = pickle.load(f)

all_lables = crys_label | NMR_lable | ETR_label

all_lables.keys()

len(all_lables.keys())

all_lables['4wisA10']=all_lables['4wisA1']
del all_lables['4wisA1']

"""## Merging Protbert embedding with THOIPA labels"""

#concatenate prtobert with the label
protbert_pluse_interface_label = {}
for key in all_protbert.keys():
  featrues= all_protbert[key]
  label = all_lables[key]
  concat= pd.concat([featrues, pd.DataFrame(label)], axis=1)
  protbert_pluse_interface_label[key] = concat

protbert_pluse_interface_label.keys()

for key in protbert_pluse_interface_label.keys():
  print(key,protbert_pluse_interface_label[key].isna().sum().sum())

protbert_pluse_interface_label['P22607']

"""## Split data into train and test dictionaries"""

# test list
test_list=['5nkqA1', '4ryiA2', '3zk1A1', '2j58A1', 'O15455', 'P20963', 'O75460', 'Q12983', 'P08514', 'P05026']
train_prot={}
test_prot={}
for key in protbert_pluse_interface_label.keys():
  if key in test_list:
    test_prot[key]=protbert_pluse_interface_label[key]
  else:
    train_prot[key]=protbert_pluse_interface_label[key]

test_prot.keys()

"""## Save train and test sets into dataframes"""

#train to dataframe
train_df=pd.DataFrame()
for key in train_prot.keys():
    train_df = pd.concat([train_df, train_prot[key]], ignore_index=True)

train_df

#test dataframe
test_df=pd.DataFrame()
for key in test_prot.keys():
    test_df = pd.concat([test_df, test_prot[key]], ignore_index=True)

test_df

# """## Train on dimers, Test on a small subset of dimers (the same test set of THOIPA)"""

# #logistic regression
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(max_iter=1000,class_weight='balanced')
# classifier.fit(train_df.iloc[:,:-1], train_df.iloc[:,-1])

# pred = classifier.predict(test_df.iloc[:,:-1])

# from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
# print("f1 score:",f1_score(test_df.iloc[:,-1], pred))
# print("precision:",precision_score(test_df.iloc[:,-1], pred))
# print("recall:",recall_score(test_df.iloc[:,-1], pred))
# print("roc_auc_score:",roc_auc_score(test_df.iloc[:,-1], pred))

"""## Using 5 fold Cross Validation"""

#all dataframe
df=pd.DataFrame()
for key in protbert_pluse_interface_label.keys():
    df = pd.concat([df, protbert_pluse_interface_label[key]], ignore_index=True)

#5 fold cross validation

scoring = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score),
    'roc_auc': make_scorer(roc_auc_score)
}

classifier = LogisticRegression(max_iter=1000,class_weight='balanced')
scores = cross_validate(classifier, df.iloc[:,:-1], df.iloc[:,-1], cv=5, scoring=scoring)
print ("\n====================================\n\t ProtBert Model Results\n====================================\n")

print("f1_score:",scores['test_f1'])
print("f1_mean:",scores['test_f1'].mean())
print("precision:",scores['test_precision'])
print("precision_mean:",scores['test_precision'].mean())
print("recall:",scores['test_recall'])
print("recall_mean:",scores['test_recall'].mean())

print ("\n====================================\n")

# scores.mean()


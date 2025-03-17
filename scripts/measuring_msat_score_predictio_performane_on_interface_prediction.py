# -*- coding: utf-8 -*-
"""Measuring_MSAT_Score_Predictio_performane_on_interface_prediction.ipynb



# Measuring_MSAT_Score_Predictio_performane_on_interface_prediction
- The goal is to compare MSAT score in predicting interface resideus in dimers
- The input is a a vector of MSAT scores representing the the contact probability of each residue in the helix and the other residues. (for example,if the dimer (helix) length is 25, we will have 25*25 scores per dimer )
- The output is 0/1 none-interface/interface
- **Here 50 dimers are used **
"""


import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score

"""## import MSA transformer scores of crystal,NMR and ETRA"""


with open('Features/MSA_Transformer_Prediction_Scores/crystal/new_processed_Dimers_MSA_Transformer_prediction_scores.pkl', 'rb') as f:
    crystal_Dimers_MSA_Transformer_prediction_scores = pickle.load(f)
with open('Features/MSA_Transformer_Prediction_Scores/NMR/processed_NMR_MSAT_score.pkl', 'rb') as f:
    NMR_Dimers_MSA_Transformer_prediction_scores = pickle.load(f)
with open('Features/MSA_Transformer_Prediction_Scores/ETRA/processed_etra_MSAT_scores.pkl', 'rb') as f:
    ETRA_Dimers_MSA_Transformer_prediction_scores = pickle.load(f)

crystal_Dimers_MSA_Transformer_prediction_scores.keys()

#rename crystal_Dimers_MSA_Transformer_prediction_scores[4wisA10] to crystal_Dimers_MSA_Transformer_prediction_scores[4wisA1]
crystal_Dimers_MSA_Transformer_prediction_scores['4wisA1'] = crystal_Dimers_MSA_Transformer_prediction_scores.pop('4wisA10')

crystal_Dimers_MSA_Transformer_prediction_scores.keys()

NMR_Dimers_MSA_Transformer_prediction_scores.keys()

ETRA_Dimers_MSA_Transformer_prediction_scores.keys()

crystal_Dimers_MSA_Transformer_prediction_scores['5nkqA3']

"""## Import interface labels of crystal,NMR and ETRA"""


with open('Features/Dimers_interface_labels/crys_proteins_interface_labels.pkl', 'rb') as f:
    crystal_THOIPA_interface_labels = pickle.load(f)
with open('Features/Dimers_interface_labels/NMR_proteins_interface_labels.pkl', 'rb') as f:
    NMR_THOIPA_interface_labels = pickle.load(f)
with open('Features/Dimers_interface_labels/ETRA_proteins_interface_labels.pkl', 'rb') as f:
    ETRA_THOIPA_interface_labels = pickle.load(f)

crystal_THOIPA_interface_labels.keys()

crystal_THOIPA_interface_labels['5nkqA1']

"""## Mergin MSAT scores and interface label"""

crys_MSAT_plus_labels={}
for key in crystal_Dimers_MSA_Transformer_prediction_scores.keys():
  features=crystal_Dimers_MSA_Transformer_prediction_scores[key].to_numpy()
  labels=crystal_THOIPA_interface_labels[key].values.reshape(-1,1)
  conn=pd.concat([pd.DataFrame(features),pd.DataFrame(labels)],axis=1)
  crys_MSAT_plus_labels[key]=conn

crys_MSAT_plus_labels.keys()

crys_MSAT_plus_labels['5nkqA1'].isna().sum().sum()

"""## Appling PCA 10 to reduce msat scores
- since MSAT scores can be of different lengths for different proteins (for example 25*25 and 28*28) we use PCA to reduce all of them to fixed length to be able to fed them to the model.

### crystal
"""

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=10)

crys_MSAT_pca10_labels={}
for key in crys_MSAT_plus_labels.keys():
  X=crys_MSAT_plus_labels[key].iloc[:,:-1]
  y=crys_MSAT_plus_labels[key].iloc[:,-1]
  pca10=pca.fit_transform(X)
  conn=pd.concat([pd.DataFrame(pca10),pd.DataFrame(y)],axis=1)
  conn.columns=['PC'+str(i) for i in range(0,11)]
  conn.rename(columns={'PC10':'label'},inplace=True)
  crys_MSAT_pca10_labels[key]=conn

crys_MSAT_pca10_labels.keys()

crys_MSAT_pca10_labels['4wisA1']

"""### NMR"""

NMR_Dimers_MSA_Transformer_prediction_scores.keys()

NMR_THOIPA_interface_labels.keys()

mapping={'2k1k':'P21709','2l6w':'P09619','2lzl':'P22607','2loh':'P05067','2l34':'O43914','2hac':'P20963','2mk9':'O15455','6nhw':'O14763'}

NMR_MSAT_plus_labels={}
for key in NMR_Dimers_MSA_Transformer_prediction_scores.keys():
  print(key)
  features=NMR_Dimers_MSA_Transformer_prediction_scores[key].to_numpy()
  labels=NMR_THOIPA_interface_labels[mapping[key[:4]]].values.reshape(-1,1)
  conn=pd.concat([pd.DataFrame(features),pd.DataFrame(labels)],axis=1)

  NMR_MSAT_plus_labels[mapping[key[:4]]]=conn

NMR_MSAT_plus_labels.keys()

for key in NMR_MSAT_plus_labels.keys():
  print(key,NMR_MSAT_plus_labels[key].isna().sum().sum())

NMR_MSAT_plus_labels['O43914']

NMR_MSAT_pca10_labels={}
for key in NMR_MSAT_plus_labels.keys():
  X=NMR_MSAT_plus_labels[key].iloc[:,:-1]
  y=NMR_MSAT_plus_labels[key].iloc[:,-1]
  pca10=pca.fit_transform(X)
  conn=pd.concat([pd.DataFrame(pca10),pd.DataFrame(y)],axis=1)
  conn.columns=['PC'+str(i) for i in range(0,11)]
  conn.rename(columns={'PC10':'label'},inplace=True)
  NMR_MSAT_pca10_labels[key]=conn

NMR_MSAT_pca10_labels.keys()

NMR_MSAT_pca10_labels['O43914']

"""### ETRA"""

ETRA_Dimers_MSA_Transformer_prediction_scores.keys()

ETRA_THOIPA_interface_labels.keys()

etra_MSAT_pca10_labels={}
for key in ETRA_Dimers_MSA_Transformer_prediction_scores.keys():
  X=ETRA_Dimers_MSA_Transformer_prediction_scores[key]
  Y=ETRA_THOIPA_interface_labels[key[:6]]
  pca10=pca.fit_transform(X)
  conn=pd.concat([pd.DataFrame(pca10),pd.DataFrame(Y)],axis=1)
  conn.columns=['PC'+str(i) for i in range(0,11)]
  conn.rename(columns={'PC10':'label'},inplace=True)
  etra_MSAT_pca10_labels[key[:6]]=conn

etra_MSAT_pca10_labels.keys()

etra_MSAT_pca10_labels['P04626']

for key in etra_MSAT_pca10_labels.keys():
  print(key,etra_MSAT_pca10_labels[key].isna().sum().sum())

"""## Merge all crystal,NMR and ETRA with interface label in one dictionary"""

all_MSAT_pca10_socres_plus_interface_labels={}
all_MSAT_pca10_socres_plus_interface_labels=crys_MSAT_pca10_labels | NMR_MSAT_pca10_labels | etra_MSAT_pca10_labels

len(all_MSAT_pca10_socres_plus_interface_labels)

all_MSAT_pca10_socres_plus_interface_labels.keys()

for key in all_MSAT_pca10_socres_plus_interface_labels.keys():
  print(key,all_MSAT_pca10_socres_plus_interface_labels[key].isna().sum().sum())

all_MSAT_pca10_socres_plus_interface_labels['4wisA1'].head()

# with open('all_MSAT_pca10_socres_plus_interface_labels.pkl', 'wb') as f:
#     pickle.dump(all_MSAT_pca10_socres_plus_interface_labels, f)

"""## Splitting data into train and test"""

test_list=['5nkqA1', '4ryiA2', '3zk1A1', '2j58A1', 'O15455', 'P20963', 'O75460', 'Q12983', 'P08514', 'P05026']

test_MSAT_pca10_socres_plus_interface_labels={}
train_MSAT_pca10_socres_plus_interface_labels={}
for key in all_MSAT_pca10_socres_plus_interface_labels.keys():
  if key in test_list:
    test_MSAT_pca10_socres_plus_interface_labels[key]=all_MSAT_pca10_socres_plus_interface_labels[key]
  else:
    train_MSAT_pca10_socres_plus_interface_labels[key]=all_MSAT_pca10_socres_plus_interface_labels[key]

train_MSAT_pca10_socres_plus_interface_labels.keys()

test_MSAT_pca10_socres_plus_interface_labels.keys()

train_MSAT_pca10_socres_plus_interface_labels_df=pd.DataFrame()
test_MSAT_pca10_socres_plus_interface_labels_df=pd.DataFrame()

for key,value in train_MSAT_pca10_socres_plus_interface_labels.items():
  print (key)
  train_MSAT_pca10_socres_plus_interface_labels_df=pd.concat([train_MSAT_pca10_socres_plus_interface_labels_df,value],axis=0,ignore_index=True)

for key,value in test_MSAT_pca10_socres_plus_interface_labels.items():
  test_MSAT_pca10_socres_plus_interface_labels_df=pd.concat([test_MSAT_pca10_socres_plus_interface_labels_df,value],axis=0,ignore_index=True)

train_MSAT_pca10_socres_plus_interface_labels_df.shape

test_MSAT_pca10_socres_plus_interface_labels_df.shape

train_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1]

"""## Model Training (logistic regression)=> train on dimers, test on a small subset."""

# logistic regression
# from sklearn.linear_model import LogisticRegression
# classifier=LogisticRegression(max_iter=1000,class_weight='balanced')
# classifier.fit(train_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,:-1],train_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1])

# pred=classifier.predict(test_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,:-1])

# from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score
# print(f1_score(test_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1],pred))
# print(precision_score(test_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1],pred))
# print(recall_score(test_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1],pred))
# print(accuracy_score(test_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1],pred))



"""## Cross Validation Using Logistic Regression on all dimers"""

all_MSAT_pca10_socres_plus_interface_labels.keys()

all_dimers_MSAT_pca10_socres_plus_interface_labels_df=pd.DataFrame()
for key,value in all_MSAT_pca10_socres_plus_interface_labels.items():
  all_dimers_MSAT_pca10_socres_plus_interface_labels_df=pd.concat([all_dimers_MSAT_pca10_socres_plus_interface_labels_df,value],axis=0,ignore_index=True)

all_dimers_MSAT_pca10_socres_plus_interface_labels_df.head()

from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,make_scorer
classifier=LogisticRegression(max_iter=1000,class_weight='balanced')
scoring={'f1_score':make_scorer(f1_score),'precision_score':make_scorer(precision_score),'recall_score':make_scorer(recall_score),'roc_auc':make_scorer(roc_auc_score)}


scores=cross_validate(classifier,all_dimers_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,:-1],all_dimers_MSAT_pca10_socres_plus_interface_labels_df.iloc[:,-1],scoring=scoring,cv=5)
scores
print ("\n====================================\n\t MSA Transformer Model Results\n====================================\n")

print("f1 scores=",scores['test_f1_score'])
print("precision scores=",scores['test_precision_score'])
print("recall scores=",scores['test_recall_score'])
print("\n")
print("mean f1 scores=",scores['test_f1_score'].mean())
print("mean precision scores=",scores['test_precision_score'].mean())
print("mean recall scores=",scores['test_recall_score'].mean())

print ("\n====================================\n")




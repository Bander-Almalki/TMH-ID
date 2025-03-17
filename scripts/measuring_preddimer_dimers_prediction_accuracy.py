# -*- coding: utf-8 -*-
"""Measuring_Preddimer_dimers_prediction_accuracy



# Measuring_Preddimer_dimers_prediction_accuracy
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score

"""## Importing Preddimer predicted Distances of Dimers (Crystal,NMR)"""


folder= 'Preddimer_predicted_dimers_structures/'

with open (os.path.join(folder,'Crystal_Preddimer_Predicted_Dimers.pkl'), 'rb') as f:
  Crystal_Preddimer_Predicted_Dimers = pickle.load(f)
with open (os.path.join(folder,'NMR_Preddimer_Predicted_Dimers.pkl'), 'rb') as f:
  NMR_Preddimer_Predicted_Dimers = pickle.load(f)

Crystal_Preddimer_Predicted_Dimers.keys()

NMR_Preddimer_Predicted_Dimers.keys()

Crystal_Preddimer_Predicted_Dimers['1orqC4']

"""## Merging Preddimer Crystal and NMR Dimers and setting a thrashold of 3.5 on Distances"""

preddimer_crystal_plus_NMR={}
preddimer_crystal_plus_NMR=Crystal_Preddimer_Predicted_Dimers | NMR_Preddimer_Predicted_Dimers
preddimer_crystal_plus_NMR.keys()

for key,value in preddimer_crystal_plus_NMR.items():
  print(key)
  dis=value['Distance']
  #thrashold of 3.5 on dis
  thr=3.5
  label=[]
  for i in range(len(dis)):
    if dis[i]<thr:
      label.append(1)
    else:
      label.append(0)
  conc=pd.concat([value,pd.DataFrame(label,columns=['Preddimer_label'])],axis=1)

  preddimer_crystal_plus_NMR[key]=conc

preddimer_crystal_plus_NMR['1orqC4']

"""## Removind extra residues from Preddimer predicted distances
- preddimer accepts only sequences that are >= 20 residues
- if the sequence is < 20 we added 5 residues before and after
- here, after predicting the distances, we need to remove those residues to be abel to compare the results with the ground truth.  
"""

preddimer_crystal_plus_NMR['4i0uA1'].shape

extra_residues_sequences=['4hksA1','4o9pC1','1xioA4','2axtM1','5nkqA1','4i0uA1','O15455']

for key,value in preddimer_crystal_plus_NMR.items():
  if key in (extra_residues_sequences):
    print(key)
    print("before deletion:",value.shape)
    new_value=value[5:-5]
    new_value.reset_index(drop=True,inplace=True)
    print("after deletion:",new_value.shape)
    preddimer_crystal_plus_NMR[key]=new_value

"""## Importing ground truth labels of Dimers (Crystal and NMR)"""



with open ('Features/Dimers_interface_labels/crys_proteins_interface_labels.pkl', 'rb') as f:
  crststal_label = pickle.load(f)
with open ('Features/Dimers_interface_labels/NMR_proteins_interface_labels.pkl', 'rb') as f:
  NMR_label = pickle.load(f)

all_labels= crststal_label | NMR_label




"""## Combin AlphaFold Predicted labels and Ground Truth labels (Crystal and NMR Dimers) to be compared"""

preddimer_prediction_plus_GTLabel={}
for key,value in preddimer_crystal_plus_NMR.items():
  print(key)
  PD_pred=value[['Residue1','Distance','Preddimer_label']]
  true_labe=all_labels[key]
  conc=pd.concat([PD_pred,true_labe],axis=1)
  conc.rename(columns={'interface': 'label'}, inplace=True)
  preddimer_prediction_plus_GTLabel[key]=conc

preddimer_prediction_plus_GTLabel['1orqC4']

"""## Calculating F1 score of each sequence individually and then the maen of all (Crystal and NMR Dimers)"""

# Sorting the dictionary by key
myKeys = list(preddimer_prediction_plus_GTLabel.keys())
myKeys.sort()
sorted_dict = {i: preddimer_prediction_plus_GTLabel[i] for i in myKeys}

print(sorted_dict.keys())

f1_list=[]
precision_list=[]
recall_list=[]
for key,value in sorted_dict.items():
  print(key)
  f1_s=f1_score(value['Preddimer_label'],value['label'])
  precision_s=precision_score(value['Preddimer_label'],value['label'])
  recall_s=recall_score(value['Preddimer_label'],value['label'])
  f1_list.append(f1_s)
  precision_list.append(precision_s)
  recall_list.append(recall_s)
  # print(f1_s)
  # print(precision_s)
  # print(recall_s)

print ("\n====================================\n\tPREDDIMER Model Results\n====================================\n")

print("f1 scores:",f1_list)
print("precision scores:",precision_s)
print("recall scores:",recall_s)


print("f1 mean score:",np.mean(f1_list)*100)

print("precision mean score:",np.mean(precision_list)*100)

print("recall mean score:",np.mean(recall_list)*100)

print ("\n====================================\n")





#bar chart f1_list
import matplotlib.pyplot as plt
from matplotlib import *
plt.bar(sorted_dict.keys(),f1_list)
plt.ylim(0, 1)
plt.xticks(rotation='vertical')
plt.xlabel('TM Homodimers')
plt.ylabel('F1_score')
plt.title('F1_score of PREDDIMER ')
plt.show()

# with open ('preddimer_prediction_plus_GTLabel.pkl', 'wb') as f:
#   pickle.dump(sorted_dict, f)



# """## Calculating the mean F1 scores of NMR dataset"""

# preddimer_prediction_plus_GTLabel.keys()

# len(preddimer_prediction_plus_GTLabel)

# NMR_list=['P20963','P21709','O43914','P05067','P09619','P22607','O15455','O14763']
# f1_list=[]
# precision_list=[]
# recall_list=[]
# for key,value in preddimer_prediction_plus_GTLabel.items():
#   if key in NMR_list:
#     f1_s=f1_score(value['Preddimer_label'],value['label'])
#     precision_s=precision_score(value['Preddimer_label'],value['label'])
#     recall_s=recall_score(value['Preddimer_label'],value['label'])
#     f1_list.append(f1_s)
#     precision_list.append(precision_s)
#     recall_list.append(recall_s)

#     print(key,f1_s)

# np.mean(f1_list)*100

# np.mean(precision_list)*100

# np.mean(recall_list)*100

# f1_list=[]
# precision_list=[]
# recall_list=[]
# for key,value in preddimer_prediction_plus_GTLabel.items():
#   if key not in NMR_list:
#     f1_s=f1_score(value['Preddimer_label'],value['label'])
#     precision_s=precision_score(value['Preddimer_label'],value['label'])
#     recall_s=recall_score(value['Preddimer_label'],value['label'])
#     f1_list.append(f1_s)
#     precision_list.append(precision_s)
#     recall_list.append(recall_s)

#     print(key,f1_s)

# np.mean(f1_list)*100

# np.mean(precision_list)*100

# np.mean(recall_list)*100


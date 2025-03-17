# -*- coding: utf-8 -*-
"""Measuring_alphafold2Mulitmer_dimers_prediction_accuracy.ipynb


# Measuring_alphafold2Mulitmer_dimers_prediction_accuracy
"""


import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
from matplotlib import *


"""## Importing AlpahFold predicted Distances of Dimers (Crystal,NMR)"""


folder= 'Alphafold2_Predicted_Dimers/'

with open (os.path.join(folder,'Crystal_Alphafold2_Predicted_Dimers.pkl'), 'rb') as f:
  Crystal_Alphafold2_Predicted_Dimers = pickle.load(f)
with open (os.path.join(folder,'NMR_Alphafold2_Predicted_Dimers.pkl'), 'rb') as f:
  NMR_Alphafold2_Predicted_Dimers = pickle.load(f)

Crystal_Alphafold2_Predicted_Dimers.keys()

NMR_Alphafold2_Predicted_Dimers.keys()

Crystal_Alphafold2_Predicted_Dimers['1orqC4']

"""## Merging AlpaFold Crystal and NMR Dimers and setting a thrashold of 3.5 on Distances"""

alphafold_crystal_plus_NMR={}
alphafold_crystal_plus_NMR=Crystal_Alphafold2_Predicted_Dimers | NMR_Alphafold2_Predicted_Dimers
alphafold_crystal_plus_NMR.keys()

#rename 4wisA10 to 4wisA1
alphafold_crystal_plus_NMR['4wisA1']=alphafold_crystal_plus_NMR.pop('4wisA10')

for key,value in alphafold_crystal_plus_NMR.items():
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
  conc=pd.concat([value,pd.DataFrame(label,columns=['AlphaFold_label'])],axis=1)

  alphafold_crystal_plus_NMR[key]=conc

alphafold_crystal_plus_NMR['1orqC4']

"""## Importing ground truth labels of Dimers (Crystal and NMR)"""




with open ('Features/Dimers_interface_labels/crys_proteins_interface_labels.pkl', 'rb') as f:
  crststal_label = pickle.load(f)
with open ('Features/Dimers_interface_labels/NMR_proteins_interface_labels.pkl', 'rb') as f:
  NMR_label = pickle.load(f)

all_labels= crststal_label | NMR_label



all_labels['1orqC4']

"""## Combin AlphaFold Predicted labels and Ground Truth labels (Crystal and NMR Dimers) to be compared"""

alphafold_prediction_plus_GTLabel={}
for key,value in alphafold_crystal_plus_NMR.items():
  print(key)
  AF_pred=value[['Residue1','Distance','AlphaFold_label']]
  true_labe=all_labels[key]
  conc=pd.concat([AF_pred,true_labe],axis=1)
  conc.rename(columns={'interface': 'label'}, inplace=True)
  alphafold_prediction_plus_GTLabel[key]=conc


"""## Calculating F1 score of each sequence individually and then the maen of all (Crystal and NMR Dimers)"""

# Sorting the dictionary
myKeys = list(alphafold_prediction_plus_GTLabel.keys())
myKeys.sort()
sorted_dict = {i: alphafold_prediction_plus_GTLabel[i] for i in myKeys}


f1_list=[]
precision_list=[]
recall_list=[]
for key,value in sorted_dict.items():
  print(key)
  f1_s=f1_score(value['AlphaFold_label'],value['label'])
  precision_s=precision_score(value['AlphaFold_label'],value['label'])
  recall_s=recall_score(value['AlphaFold_label'],value['label'])
  f1_list.append(f1_s)
  precision_list.append(precision_s)
  recall_list.append(recall_s)
  # print(f1_s)
  # print(precision_s)
  # print(recall_s)

print ("\n====================================\n\tAlphaFold2Multimer Model Results\n====================================\n")

print("f1 scores:",f1_list)
print("precision scores:",precision_s)
print("recall scores:",recall_s)


print("f1 mean score:",np.mean(f1_list)*100)

print("precision mean score:",np.mean(precision_list)*100)

print("recall mean score:",np.mean(recall_list)*100)

print ("\n====================================\n")

#bar chart f1_list

plt.bar(sorted_dict.keys(),f1_list)
plt.ylim(0,1)
plt.title('F1_score of Alphafold2_Multimer')
plt.xticks(rotation='vertical')
plt.xlabel('TM Homodimers')
plt.ylabel('F1_score')
plt.show()

# with open('alphafold_prediction_plus_GTLabel.pkl', 'wb') as f:
#   pickle.dump(sorted_dict, f)


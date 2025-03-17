# -*- coding: utf-8 -*-
"""Measuring_Thoipa_performance_on_interface_prediction.ipynb


"""

# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

"""## Importing Thoipa Data (features and labels)"""


Thoipa_crystal_prots={}
folder='THOIPA/Data/features/combined/crystal'
for file in os.listdir(folder):
  if file.endswith('.csv'):
    print(file)
    df=pd.read_csv(os.path.join(folder,file))
    df.drop(columns=['Unnamed: 0',],inplace=True)
    Thoipa_crystal_prots[file[:6]]=df

Thoipa_crystal_prots.keys()

Thoipa_crystal_prots['1xioA4'].columns

Thoipa_NMR_prots={}
folder='THOIPA/Data/features/combined/NMR'
for file in os.listdir(folder):
  if file.endswith('.csv'):
    print(file)
    df=pd.read_csv(os.path.join(folder,file))
    df.drop(columns=['Unnamed: 0',],inplace=True)
    Thoipa_NMR_prots[file[:6]]=df

Thoipa_NMR_prots.keys()

Thoipa_NMR_prots['P21709'].columns

Thoipa_ETRA_prots={}
folder='THOIPA/Data/features/combined/ETRA'
for file in os.listdir(folder):
  if file.endswith('.csv'):
    print(file)
    df=pd.read_csv(os.path.join(folder,file))
    df.drop(columns=['Unnamed: 0',],inplace=True)
    Thoipa_ETRA_prots[file[:6]]=df

Thoipa_ETRA_prots.keys()

Thoipa_ETRA_prots['P02724'].columns.tolist()

all_thoipa_prots={}
all_thoipa_prots.update(Thoipa_crystal_prots)
all_thoipa_prots.update(Thoipa_NMR_prots)
all_thoipa_prots.update(Thoipa_ETRA_prots)

all_thoipa_prots.keys()

len(all_thoipa_prots.keys())

"""## Spliting data into train and test"""

test_list=['2j58A1', '3zk1A1', '4ryiA2', '5nkqA1','P20963', 'O15455', 'O75460', 'P08514', 'Q12983','P05026']

train_data={}
test_data={}
for key in all_thoipa_prots.keys():
  if key not in test_list:
    train_data[key]=all_thoipa_prots[key]
  else:
    test_data[key]=all_thoipa_prots[key]

train_data.keys()

test_data.keys()

train_features=pd.DataFrame()
train_labels=pd.DataFrame()
for key,value in train_data.items():
  x=value.drop(columns=['residue_name', 'residue_num','interface_score','interface','interface_score_norm'],axis=1)
  y=value['interface'].astype('int')
  train_features=pd.concat([train_features,x],axis=0)
  train_labels=pd.concat([train_labels.astype(int),y],axis=0)

train_features.head()

train_features.shape

train_labels.shape

train_labels.isnull().sum()

train_labels.value_counts()

test_features=pd.DataFrame()
test_labels=pd.DataFrame()
for key,value in test_data.items():
  x=value.drop(columns=['residue_name', 'residue_num','interface_score','interface','interface_score_norm'],axis=1)
  y=value['interface']
  test_features=pd.concat([test_features,x],axis=0)
  test_labels=pd.concat([test_labels.astype(int),y],axis=0)

test_features.shape

test_labels.shape

test_labels.value_counts()

# """## Train on 40 dimers Test on 10 dimers (interface prediction task)

# ### Logistic Regression
# """

# # logistic regression classifier
# from sklearn.linear_model import LogisticRegression
# classifier=LogisticRegression(max_iter=10000,class_weight='balanced')
# classifier.fit(train_features,train_labels.values.ravel())

# pred=classifier.predict(test_features)

# from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
# print('f1_score:',f1_score(test_labels,pred))
# print('precision:',precision_score(test_labels,pred))
# print('recall:',recall_score(test_labels,pred))
# print('ROC_AUC:',roc_auc_score(test_labels,pred))

# # ploting precision recall curve
# from sklearn.metrics import precision_recall_curve
# import matplotlib.pyplot as plt
# precision, recall, thresholds = precision_recall_curve(test_labels, pred)
# plt.plot(recall, precision)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.show()

# """### Extra Trees Classifier"""

# from sklearn.ensemble import ExtraTreesClassifier
# classifier=ExtraTreesClassifier()
# classifier.fit(train_features,train_labels.values.ravel())

# pred2=classifier.predict(test_features)

# print('f1_score:',f1_score(test_labels,pred2))
# print('precision:',precision_score(test_labels,pred2))
# print('recall:',recall_score(test_labels,pred2))
# print('ROC_AUC:',roc_auc_score(test_labels,pred2))

"""## 5 Fold Cross validation

### Logistic Regression
"""

all_featues=pd.concat([train_features,test_features],axis=0)
all_labels=pd.concat([train_labels,test_labels],axis=0)

all_featues.shape

all_labels.shape

# 5 fold cross validation
# classifier=LogisticRegression(max_iter=10000,class_weight='balanced')

classifier2=LogisticRegression(max_iter=10000,class_weight='balanced')
scores2 = cross_validate(classifier2, all_featues, all_labels.values.ravel(), cv=5,scoring=['f1','precision', 'recall','roc_auc'])
print ("\n====================================\n\tResults\n====================================\n")
print('f1_score:',scores2['test_f1'])
print('precision:',scores2['test_precision'])
print('recall:',scores2['test_recall'])
print("\n")
print('f1_scores_mean:',np.mean(scores2['test_f1']))
print('precision_scores_mean:',np.mean(scores2['test_precision']))
print('recall_scores_mean:',np.mean(scores2['test_recall']))

print ("\n====================================\n")
# """### Extra trees classifier"""

# extra_classifier=ExtraTreesClassifier()
# scores3 = cross_validate(extra_classifier, all_featues, all_labels.values.ravel(), cv=5,scoring=['f1','precision', 'recall','roc_auc'])

# print('f1_score:',scores3['test_f1'])
# print('precision:',scores3['test_precision'])
# print('recall:',scores3['test_recall'])
# print('ROC_AUC:',scores3['test_roc_auc'])

# print('f1_scores_mean:',np.mean(scores3['test_f1']))
# print('precision_scores_mean:',np.mean(scores3['test_precision']))
# print('recall_scores_mean:',np.mean(scores3['test_recall']))
# print('ROC_AUC_scores_mean:',np.mean(scores3['test_roc_auc']))


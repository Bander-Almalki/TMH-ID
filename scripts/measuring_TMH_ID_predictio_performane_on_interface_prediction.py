# -*- coding: utf-8 -*-
"""Merging_MSAT_Lips_Motifs_PreddimerDistance


"""


import os
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,make_scorer



"""## Importing all_MSAT_pca10_plus_lips_motifs_scores"""


with open('Features/Combined/all_MSAT_pca10_plus_lips_motifs_scores.pkl', 'rb') as f:
    all_MSAT_pca10_plus_lips_motifs_scores = pickle.load(f)

all_MSAT_pca10_plus_lips_motifs_scores['1orqC4']

all_MSAT_pca10_plus_lips_motifs_scores.keys()

"""## Importing Preddimer predicted distances"""


with open('Preddimer_predicted_dimers_structures/Crystal_Preddimer_Predicted_Dimers.pkl', 'rb') as f:
    Crystal_Preddimer_Predicted_Dimers = pickle.load(f)
with open('Preddimer_predicted_dimers_structures/NMR_Preddimer_Predicted_Dimers.pkl', 'rb') as f:
    NMR_Preddimer_Predicted_Dimers = pickle.load(f)
with open('Preddimer_predicted_dimers_structures/ETRA_Preddimer_Predicted_Dimers.pkl', 'rb') as f:
    ETRA_Preddimer_Predicted_Dimers = pickle.load(f)

all_Preddimer_Prediction= Crystal_Preddimer_Predicted_Dimers | NMR_Preddimer_Predicted_Dimers | ETRA_Preddimer_Predicted_Dimers

all_Preddimer_Prediction.keys()

all_Preddimer_Prediction['1orqC4']

"""## Removind extra residues from Preddimer predicted distances
- preddimer accepts only sequences that are >= 20 residues
- if the sequence is < 20 we added 5 residues before and after
- here, after predicting the distances, we need to remove those residues to be abel to compare the results with the ground truth.  
"""

all_Preddimer_Prediction['4i0uA1'].shape

extra_residues_sequences=['4hksA1','4o9pC1','1xioA4','2axtM1','5nkqA1','4i0uA1','O15455','P02724','P05106','P0A6S5','Q12983','Q6ZRP7','Q8NI60','Q99IB8']

for key,value in all_Preddimer_Prediction.items():
  if key in (extra_residues_sequences):
    if key=='P0A6S5':
      new_value=value[4:-5]
      new_value.reset_index(drop=True,inplace=True)
      all_Preddimer_Prediction[key]=new_value
    else:

      print(key)
      print("before deletion:",value.shape)
      new_value=value[5:-5]
      new_value.reset_index(drop=True,inplace=True)
      print("after deletion:",new_value.shape)
      all_Preddimer_Prediction[key]=new_value

for key,value in all_Preddimer_Prediction.items():
  print(key)
  print(value.shape)
  print(all_MSAT_pca10_plus_lips_motifs_scores[key].shape)

"""## Merging all_MSAT_pca10_plus_lips_motifs_scores, Preddimer predicted distances"""

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance={}
for key,value in all_MSAT_pca10_plus_lips_motifs_scores.items():
    conn=pd.concat([value,all_Preddimer_Prediction[key]['Distance']],axis=1)
    all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance[key]=conn

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.keys()

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance['1orqC4']

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df=pd.DataFrame()
for key,value in all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.items():
    all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df=pd.concat([all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df,value],axis=0)

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df

all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df.columns

"""## Logistic Regression trained on dimers(40 dimers) and tested on a small dimer set (10 dimers)"""

test_list=['5nkqA1', '4ryiA2', '3zk1A1', '2j58A1', 'O15455', 'P20963', 'O75460', 'Q12983', 'P08514', 'P05026']
train_set={}
test_set={}
for key,value in all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.items():
    if key not in test_list:
        train_set[key]=value
    else:
        test_set[key]=value

test_set.keys()

train_df=pd.DataFrame()
test_df=pd.DataFrame()
for key,value in train_set.items():
    train_df=pd.concat([train_df,value],axis=0)
for key,value in test_set.items():
    test_df=pd.concat([test_df,value],axis=0)

x_train=train_df.drop(['label'],axis=1)
y_train=train_df['label']
x_test=test_df.drop(['label'],axis=1)
y_test=test_df['label']

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score
# classifier=LogisticRegression(max_iter=1000,class_weight='balanced')
# classifier.fit(x_train,y_train)
# y_pred=classifier.predict(x_test)
# print('f1_score=',f1_score(y_test,y_pred))
# print('precision score=',precision_score(y_test,y_pred))
# print('recall score=',recall_score(y_test,y_pred))
# print('roc_auc score=',roc_auc_score(y_test,y_pred))

# classifier.coef_

# classifier.intercept_

# # rank classifier coef_
# coef_df=pd.DataFrame(classifier.coef_,columns=x_train.columns)
# coef_df

"""## Cross-validation"""

# 5fold cross-validation

scoring={'f1_score':make_scorer(f1_score),'precision_score':make_scorer(precision_score),'recall_score':make_scorer(recall_score),'roc_auc_score':make_scorer(roc_auc_score)}
classifier=LogisticRegression(max_iter=1000,class_weight='balanced')
X_all=all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df.drop('label',axis=1)
y_all=all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df['label']
scores=cross_validate(classifier, X_all,y_all,cv=5, scoring=scoring, return_estimator=True)
feature_weights = [est.coef_[0] for est in scores['estimator']]
# print(scores)


print ("\n====================================\n\tTMH-ID Model Results\n====================================\n")

print('f1_score=',scores['test_f1_score'])
print('precision score=',scores['test_precision_score'])
print('recall score=',scores['test_recall_score'])
print("\n")
print('mean f1 score=',np.mean(scores['test_f1_score']))
print('mean precision score=',np.mean(scores['test_precision_score']))
print('mean recall score=',np.mean(scores['test_recall_score']))

print ("\n====================================\n")

feature_weights

# """## Measuring the contribution of the PREDDIMER distance alone to predict the interface residue"""

# all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df.head()

# scoring={'f1_score':make_scorer(f1_score),'precision_score':make_scorer(precision_score),'recall_score':make_scorer(recall_score),'roc_auc_score':make_scorer(roc_auc_score)}
# classifier=LogisticRegression(max_iter=1000,class_weight='balanced')
# X=all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df[['Distance']]
# y=all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance_df['label']
# scores=cross_validate(classifier,X,y,cv=5,scoring=scoring)
# print(scores)

# print('f1_score=',scores['test_f1_score'])
# print('mean f1 score=',np.mean(scores['test_f1_score']))



# """## Training on ETRA and NMR, Testing on Crystal

# """

# all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.keys()

# test_list2=['1orqC4','1xioA4','2axtM1','2h8aA2','2j58A1','3h9vA2','3rifA2','3spcA2','3zk1A1','4hksA1',
#             '4i0uA1','4o9pC1','4r0cA7','4ryiA2','4wisA1','5irzD6','5nkqA1','5nkqA3','5t4dA6','5u6oA6','5uldA9']
# train_set2={}
# test_set2={}
# for key,value in all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.items():
#     if key not in test_list2:
#         train_set2[key]=value
#     else:
#         test_set2[key]=value

# train_set2.keys()

# len(train_set2)

# test_set2.keys()

# train_df2=pd.DataFrame()
# test_df2=pd.DataFrame()
# for key,value in train_set2.items():
#     train_df2=pd.concat([train_df2,value],axis=0)
# for key,value in test_set2.items():
#     test_df2=pd.concat([test_df2,value],axis=0)

# x_train2=train_df2.drop(['label'],axis=1)
# y_train2=train_df2['label']
# x_test2=test_df2.drop(['label'],axis=1)
# y_test2=test_df2['label']

# classifier2=LogisticRegression(max_iter=1000,class_weight='balanced')
# classifier2.fit(x_train2,y_train2)
# y_pred2=classifier2.predict(x_test2)
# print('f1_score=',f1_score(y_test2,y_pred2))
# print('precision score=',precision_score(y_test2,y_pred2))
# print('recall score=',recall_score(y_test2,y_pred2))
# print('roc_auc score=',roc_auc_score(y_test2,y_pred2))

# """## Training on ETRA and Crystal, testing on NMR"""

# test_list3=['P20963','P21709','O43914','P05067','P09619','P22607','O15455','O14763']
# train_set3={}
# test_set3={}
# for key,value in all_MSAT_pca10_plus_lips_motifs_scores_Preddimer_Distance.items():
#     if key not in test_list3:
#         train_set3[key]=value
#     else:
#         test_set3[key]=value

# len(train_set3)

# train_set3.keys()

# test_set3.keys()

# train_df3=pd.DataFrame()
# test_df3=pd.DataFrame()
# for key,value in train_set3.items():
#     train_df3=pd.concat([train_df3,value],axis=0)
# for key,value in test_set3.items():
#     test_df3=pd.concat([test_df3,value],axis=0)

# x_train3=train_df3.drop(['label'],axis=1)
# y_train3=train_df3['label']
# x_test3=test_df3.drop(['label'],axis=1)
# y_test3=test_df3['label']

# classifier3=LogisticRegression(max_iter=1000,class_weight='balanced')
# classifier3.fit(x_train3,y_train3)
# y_pred3=classifier3.predict(x_test3)
# print('f1_score=',f1_score(y_test3,y_pred3))
# print('precision score=',precision_score(y_test3,y_pred3))
# print('recall score=',recall_score(y_test3,y_pred3))
# print('roc_auc score=',roc_auc_score(y_test3,y_pred3))

# """## Using Neural Networks"""

# # a neural network model with 3 layers
# from keras.models import Sequential
# from keras.layers import Dense,Dropout
# from sklearn.utils.class_weight import compute_class_weight

# model=Sequential()
# model.add(Dense(10,input_dim=17,activation='relu'))
# #add droput
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# # Calculate class weights
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weights_dict = dict(enumerate(class_weights))

# print(class_weights_dict)

# x_train = x_train.to_numpy() if isinstance(x_train, pd.Series) else x_train
# y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train


# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'],)
# model.fit(x_train,y_train,epochs=100,batch_size=100,class_weight=class_weights_dict)

# from sklearn.metrics import f1_score, classification_report
# for i in range (1,90,5):
#   print(i/100)
#   y_pred = (model.predict(x_test) > i/100).astype("int32")

#   # Calculate F1 score
#   f1 = f1_score(y_test, y_pred)
#   print("F1 Score:", f1)

# """## NN with Cross validation"""

# !pip install scikeras

# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold,cross_validate

# from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,make_scorer
# scoring={'f1_score':make_scorer(f1_score),'precision_score':make_scorer(precision_score),'recall_score':make_scorer(recall_score),'roc_auc_score':make_scorer(roc_auc_score)}


# # Function to create model, required for KerasClassifier
# def create_model():
#     # Create the model
#     model = Sequential()
#     model.add(Dense(10, input_dim=17, activation='relu'))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))

#     # Compile the model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     return model

# print(X_all.shape)
# print(y_all.shape)

# # Create the KerasClassifier wrapper
# model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100)

# # Define the cross-validation procedure
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # Perform cross-validation
# results =cross_val_score(model,X_all,y_all,cv=kfold,scoring='f1')

# #cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# # Print cross-validation results
# print(f"Cross-validation f1 scores: {results}")
# print(f"Mean f1: {results.mean():.4f}, Standard deviation: {results.std():.4f}")

# sum(y_all==1)

# y_all.shape[0] - sum(y_all==1)


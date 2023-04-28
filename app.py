import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV


######## data manipulation
df = pd.read_csv("secondary_data.csv", index_col=0, sep=";")
df = df.reset_index()
df = df.drop(['stem-root', 'veil-type', 'veil-color', 'spore-print-color', 'stem-surface'], axis = 1)
df = df.drop(['gill-spacing', 'gill-attachment', 'cap-surface'], axis = 1)
df['class'] = df['class'].replace({'p': 1, 'e': 0})
df['has-ring'] = df['has-ring'].replace({'t': 1, 'f': 0})
df['does-bruise-or-bleed'] = df['does-bruise-or-bleed'].replace({'t': 1, 'f': 0})
df = pd.get_dummies(df, columns=['season'])
df = pd.get_dummies(df, columns = ['habitat'])
df = pd.get_dummies(df, columns = ['cap-shape'])
df = pd.get_dummies(df, columns = ['cap-color'])
df = pd.get_dummies(df, columns = ['gill-color'])
df = pd.get_dummies(df, columns = ['stem-color'])
df = pd.get_dummies(df, columns = ['ring-type'])
df = df.drop(['ring-type_f'], axis = 1)
df = df.drop('stem-width', axis = 1)
df['cap-diameter_1'] = df['cap-diameter'].apply(lambda x: 'Small' if x <= 5 else 'Large' if x >= 7.5 else 'Medium')
df['stem-height'] = df['stem-height'].apply(lambda x: 'Small' if x <= 5 else 'Large' if x >= 7.5 else 'Medium')
df = pd.get_dummies(df, columns = ['cap-diameter_1'])
#df = pd.get_dummies(df, columns = ['stem-width_1'])
df = pd.get_dummies(df, columns = ['stem-height'])
df = df.drop(['cap-diameter'], axis = 1)

##########streamlit inputs###############













### beginning of model ###
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#t1 = datetime.datetime.now()
base_models = [('knn1', KNeighborsClassifier(n_neighbors=11, weights = 'distance')),
               ('knn2', KNeighborsClassifier(n_neighbors=13, weights = 'distance')),
               ('knn3', KNeighborsClassifier(n_neighbors=7, weights = 'distance'))]
meta_model = LogisticRegression()
stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model)
for name, model in base_models:
    model.fit(X_train, y_train)
base_predictions = np.column_stack([model.predict_proba(X_test)[:,1] for name, model in base_models])
meta_model.fit(base_predictions, y_test)
#t2 = datetime.datetime.now()
#print(t2-t1)
stacking_predictions = meta_model.predict(base_predictions)
stacking_accuracy_hp1 = accuracy_score(y_test, stacking_predictions)

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix 
from sklearn.ensemble import StackingClassifier



#st.write("hello world")
### Data input ###
df = pd.read_csv("secondary_data.csv", index_col = 0, sep=";")


#st.write(df.tail())
st.header('Mushroom Poison Predictor')
st.write('Please select the features your mushroom has below')
st.write('when completed select "run model" to see if your mushroom is poisonous')
st.write('Safe foragaing!')
df = df.reset_index()
df = df.drop('stem-root', axis=1)
df = df.drop('veil-type', axis=1)
df = df.drop('veil-color', axis=1)
df = df.drop('spore-print-color', axis=1)
df = df.drop('stem-surface', axis=1)
df = df.drop('gill-spacing', axis=1)
df = df.drop('gill-attachment', axis=1)
df = df.drop('cap-surface', axis=1)
#df = df.drop(['stem-root', 'veil-type', 'veil-color', 'spore-print-color', 'stem-surface'], axis = 1)
#df = df.drop(['gill-spacing', 'gill-attachment', 'cap-surface'], axis = 1)
#st.write('check_1')
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
df = df.drop('ring-type_f', axis = 1)
df = df.drop('stem-width', axis = 1)
#st.write('check_2')
df['cap-diameter_1'] = df['cap-diameter'].apply(lambda x: 'Small' if x <= 5 else 'Large' if x >= 7.5 else 'Medium')
df['stem-height'] = df['stem-height'].apply(lambda x: 'Small' if x <= 5 else 'Large' if x >= 7.5 else 'Medium')
df = pd.get_dummies(df, columns = ['cap-diameter_1'])
#df = pd.get_dummies(df, columns = ['stem-width_1'])
df = pd.get_dummies(df, columns = ['stem-height'])
df = df.drop(['cap-diameter'], axis = 1)
#st.write('check_3')
df_i = df.copy().iloc[0:0]
df_i.loc[0] = 0
df_i = df_i.drop('class', axis = 1)

counts = df_i.nunique()
pd.set_option('display.max_rows', None)
#st.write(counts)
#st.write((df_i))
st.image('Mushroom.png')

### st inputs
col1, col2 = st.columns(2)
with col1:
	seas = st.selectbox(label = 'Season', options = ['Spring', 'Summer', 'Fall', 'Winter'], index = 0)
	hab = st.selectbox(label = 'Habitat', options = ['Grass', 'Leaves', 'Meadows', 'Paths', 'Heaths', 'Urban', 'Waste', 'Woods'], index = 0)
	cs = st.selectbox(label = 'Cap Shape', options = ['Bell', 'Conical', 'Convex', 'Flat', 'Sunken', 'Spherical', 'Other'], index = 0)
	cc = st.selectbox(label = 'Cap Color', options = ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple', 'Red', 'White', 'Yellow', 'Blue', 'Orange', 'Black'], index = 0)
	gc = st.selectbox(label = 'Gill Color', options = ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple', 'Red', 'White', 'Yellow', 'Blue', 'Orange', 'Black', 'None'], index = 0)
with col2:
	sc = st.selectbox(label = 'Stem Color', options = ['Brown', 'Buff', 'Gray', 'Green', 'Pink', 'Purple', 'Red', 'White', 'Yellow', 'Blue', 'Orange', 'Black', 'None'], index = 0)
	rt = st.selectbox(label = 'Ring Type', options = ['Evanescant', 'Flaring', 'Grooved', 'Large', 'Pendant', 'Sheathing', 'Zone', 'Scaly', 'Movable', 'None'], index = 0)
	cd = st.selectbox(label = 'Cap Diameter', options = ['Smaller than the diameter of a soda can', 'Larger than a baseball', 'between these'], index = 0)
	sh = st.selectbox(label = 'Stem Height', options = ['Smaller than the diameter of a soda can', 'Larger than a baseball', 'between these'], index = 0)
	bb = st.radio(label = 'Bruise or Bleed', options = ['Yes', 'No'] )


#### Does bruise or bleed
if bb == 'Yes':
	df_i['does-bruise-or-bleed'] = 1

### cap diameter 

if cd == 'Smaller than the diameter of a soda can':
	df_i["cap-diameter_1_Small"] = 1
if cd == 'between these':
	df_i["cap-diameter_1_Medium"] = 1
if cd == 'Larger than a baseball':
	df_i["cap-diameter_1_Large"] = 1

### stem height

if sh == 'Smaller than the diameter of a soda can':
	df_i["stem-height_Small"] = 1
if sh == 'between these':
	df_i["stem-height_Medium"] = 1
if sh == 'Larger than a baseball':
	df_i["stem-height_Large"] = 1

### ring type inputs


if rt == 'Evanescant':
	df_i["ring-type_e"] = 1
if rt == 'Flaring':
	df_i["ring-type_f"] = 1
if rt == 'Grooved':
	df_i["ring-type_g"] = 1
if rt == 'Pendant':
	df_i["ring-type_p"] = 1
if rt == 'Large':
	df_i["ring-type_l"] = 1
if rt == 'Zone':
	df_i["ring-type_z"] = 1
if rt == 'Scaly':
	df_i["ring-type_y"] = 1
if rt == 'Movable':
	df_i["ring-type_m"] = 1
if rt == 'None':
	df_i['has-ring'] = 0
else:
	df_i['has-ring'] = 1



### Cap color input
if cc == 'Brown':
	df_i["cap-color_n"] = 1
if cc == 'Buff':
	df_i["cap-color_b"] = 1
if cc == 'Green':
	df_i["cap-color_r"] = 1
if cc == 'Gray':
	df_i["cap-color_g"] = 1
if cc == 'Pink':
	df_i["cap-color_p"] = 1
if cc == 'Purple':
	df_i["cap-color_u"] = 1
if cc == 'Red':
	df_i["cap-color_e"] = 1
if cc == 'White':
	df_i["cap-color_w"] = 1
if cc == 'Yellow':
	df_i["cap-color_y"] = 1
if cc == 'Blue':
	df_i["cap-color_l"] = 1
if cc == 'Orange':
	df_i["cap-color_o"] = 1
if cc == 'Black':
	df_i["cap-color_k"] = 1

### gill color input
if gc == 'Brown':
	df_i["gill-color_n"] = 1
if gc == 'Buff':
	df_i["gill-color_b"] = 1
if gc == 'Green':
	df_i["gill-color_r"] = 1
if gc == 'Gray':
	df_i["gill-color_g"] = 1
if gc == 'Pink':
	df_i["gill-color_p"] = 1
if gc == 'Purple':
	df_i["gill-color_u"] = 1
if gc == 'Red':
	df_i["gill-color_e"] = 1
if gc == 'White':
	df_i["gill-color_w"] = 1
if gc == 'Yellow':
	df_i["gill-color_y"] = 1
if gc == 'Blue':
	df_i["gill-color_l"] = 1
if gc == 'Orange':
	df_i["gill-color_o"] = 1
if gc == 'Black':
	df_i["gill-color_k"] = 1
if gc == 'None':
	df_i['gill-color_f'] = 1

### stem color input
if sc == 'Brown':
	df_i["gill-color_n"] = 1
if sc == 'Buff':
	df_i["gill-color_b"] = 1
if sc == 'Green':
	df_i["gill-color_r"] = 1
if sc == 'Gray':
	df_i["gill-color_g"] = 1
if sc == 'Pink':
	df_i["gill-color_p"] = 1
if sc == 'Purple':
	df_i["gill-color_u"] = 1
if sc == 'Red':
	df_i["gill-color_e"] = 1
if sc == 'White':
	df_i["gill-color_w"] = 1
if sc == 'Yellow':
	df_i["gill-color_y"] = 1
if sc == 'Blue':
	df_i["gill-color_l"] = 1
if sc == 'Orange':
	df_i["gill-color_o"] = 1
if sc == 'Black':
	df_i["gill-color_k"] = 1
if sc == 'None':
	df_i['gill-color_f'] = 1



### season input
if seas == 'Spring':
	df_i["season_s"] = 1
if seas == 'Summer':
	df_i["season_u"] = 1
if seas == 'Fall':
	df_i["season_a"] = 1
if seas == 'Winter':
	df_i["season_w"] = 1


### habitat input 
if hab == 'Grass':
	df_i["habitat_g"] = 1
if hab == 'Leaves':
	df_i["habitat_l"] = 1
if hab == 'Meadows':
	df_i["habitat_m"] = 1
if hab == 'Paths':
	df_i["habitat_p"] = 1
if hab == 'Heaths':
	df_i["habitat_h"] = 1
if hab == 'Urban':
	df_i["habitat_u"] = 1
if hab == 'Waste':
	df_i["habitat_w"] = 1
if hab == 'Woods':
	df_i["habitat_d"] = 1

### cap shape input
if cs == 'Bell':
	df_i["cap-shape_b"] = 1
if cs == 'Conical':
	df_i["cap-shape_c"] = 1
if cs == 'Flat':
	df_i["cap-shape_f"] = 1
if cs == 'Convex':
	df_i["cap-shape_x"] = 1
if cs == 'Sunken':
	df_i["cap-shape_s"] = 1
if cs == 'Spherical':
	df_i["cap-shape_p"] = 1
if cs == 'Other':
	df_i["cap-shape_o"] = 1


if st.button('run model'):
	base_models = [('knn1', KNeighborsClassifier(n_neighbors=11, weights='distance')),               ('knn2', KNeighborsClassifier(n_neighbors=13, weights='distance')),               ('knn3', KNeighborsClassifier(n_neighbors=7, weights='distance'))]
	meta_model = LogisticRegression()
	st.write('1')
	stacking = StackingClassifier(estimators=base_models, final_estimator=meta_model)
	X_train = df.drop('class', axis=1)
	st.write('2')
	y_train = df['class']
	for name, model in base_models:
	    model.fit(X_train, y_train)
	base_predictions = np.column_stack([model.predict_proba(X_train)[:, 1] for name, model in base_models])
	st.write('3')
	meta_model.fit(base_predictions, y_train)
	input_observation = df_i
	input_predictions = meta_model.predict(np.column_stack([model.predict_proba(input_observation)[:, 1] for name, model in base_models]))
	if input_predictions == 1:

		st.image('bad.png')
		st.write('This mushroom is dangerous. Do not consume. If you have already consumed this mushroom please contact poison control at 800-222-1222')
	else:
		st.image('good.png')
		st.write("Enjoy the mushroom!")
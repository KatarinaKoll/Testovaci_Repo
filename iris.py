# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd

#print(os.getcwd())

path = 'C:\\Users\\kkollarova\\Desktop\\ine\\Kurz_Data_science\\KURZY_Skillmea\\Zaklady_Machine_learning_v_Pythone\\Data'
os.chdir (path)
print('Aktualne WS je: '+os.getcwd())

iris = pd.read_csv('IRIS.csv')

# iris.info()

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
iris['species_enc'] = label_encoder.fit_transform(iris['species'])


y = iris['species_enc'].values
x = iris[['petal_length','petal_width']].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
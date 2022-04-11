import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

data=pd.read_csv('data.csv')
#data.head(90)

data.drop('id',axis=1,inplace=True)
data.drop('Unnamed: 32',axis=1,inplace=True)
#data.head()
data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})
#data.info()


labels=data['diagnosis']
features=data.iloc[:, 1:]


features=pd.get_dummies(features)
x_train, x_test, y_train, y_test= train_test_split(features, labels, test_size=0.3, random_state=42)

ct=ColumnTransformer([('scale',StandardScaler(),features.columns)])
x_train=ct.fit_transform(x_train)
x_test=ct.transform(x_test)


model=LogisticRegression()

model.fit(x_train, y_train)
print('train part score --> ',model.score(x_train, y_train))
print('test part score -->', model.score(x_test, y_test))

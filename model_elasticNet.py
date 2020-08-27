# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:39:29 2018

@author: 231469242@qq.com
微信公众号：pythonEducation

original_data
MAE 43.382696283539964
score: 0.022531303497459347
Variance score: 0.44

processed_data
MAE 63.93590233858747
score: 0.015399801403941694
Variance score: 0.00
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

readFileName="original_data.xlsx"
#readFileName="processed_data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.loc[:,"AGE":"S6"]
y=data["y"]

#划分训练集和测试集
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X_transform, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = ElasticNet()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
#rms = (np.mean((y - y_pred)**2))**0.5
MAE=sum(abs(y_test - y_pred))/len(y_test)
score=1/(1+MAE)
#print ("RF RMS", rms)
print("MAE",MAE)
print("score:",score)
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))



# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 09:30:24 2018

@author: 231469242@qq.com
微信公众号：pythonEducation

Age(年龄)、性别(Sex)、Body mass index(体质指数)、Average Blood Pressure(平均血压)、

S1~S6一年后疾病级数指标

original_data
linear_model.LinearRegression()
Coefficients: 
 [  47.74657117 -241.99180361  531.96856896  381.56529922 -918.49020552
  508.25147385  116.94040498  269.48508571  695.8062205    26.32343144]
MAE 41.54836328325207
r2: 0.48

processed_data
Coefficients: 
 [  29.25034582 -261.70768053  546.29737263  388.40077257 -901.95338706
  506.761149    121.14845948  288.02932495  659.27133846   41.37536901]
MAE 41.91925360556678
r2: 0.47729
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,median_absolute_error
from sklearn.model_selection import train_test_split

readFileName="processed_data.xlsx"
#读取excel
data=pd.read_excel(readFileName)
X=data.loc[:,"age":"s6"]
y=data['target']

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)
#平均绝对误差
MAE=sum(abs(y_test - y_pred))/len(y_test)
MAE1=mean_absolute_error(y_test,y_pred)
#中值绝对误差
MedianAE=median_absolute_error(y_test,y_pred) 
r2=r2_score(y_test, y_pred)

dict1={"y_predict":y_pred,"y_test":y_test}
df1=pd.DataFrame(dict1)
#df1.to_excel("MAE.xlsx")





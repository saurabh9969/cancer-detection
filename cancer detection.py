# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 14:19:00 2018

@author: saurabh
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
import statsmodels.formula.api as smf

os.getcwd()

os.chdir("C:\\Users\\saurabh\\Downloads")

data=pd.read_csv("C:\\Users\\saurabh\\Downloads\\New folder\\data.csv")

data.describe()
data.isnull().any()
data=data.drop("Unnamed: 32",axis=1)

df=data
df.head()
x=pd.DataFrame(data.drop("diagnosis",axis=1))
y=pd.DataFrame(data.diagnosis)
y.diagnosis=pd.get_dummies(y.diagnosis)

#vif
model=pd.DataFrame(x)
vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model.values,i)for i in range (model.shape[1])]
vif["feature"]=model.columns
vif.round(1)
m=smf.OLS(y,model).fit()
m.summary()

model1=model

model1=model1.drop("fractal_dimension_mean",axis=1)

#model1 vif and correlations

vif=pd.DataFrame()
vif['vif factor']=[variance_inflation_factor(model1.values,i)for i in range (model1.shape[1])]
vif['feature']=model1.columns
vif.round(2)
m=smf.OLS(y,model1).fit()
m.summary()

model2=model
model2=model2.drop("id",axis=1)

#model2 vif and correlations

vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model2.values,i)for i in range(model2.shape[1])]
vif["feature"]=model2.columns
vif.round(3)
m=smf.OLS(y,model2).fit()
m.summary()


model3=model2

model3=model3.drop("compactness_se",axis=1)

#model3 vif and correlation

vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model3.values,i)for i in range(model3.shape[1])]
vif["feature"]=model3.columns
vif.round(4)
m=smf.OLS(y,model3).fit()
m.summary()

model4=model3

model4=model4.drop("fractal_dimension_se",axis=1)

#model4 vif and correlation
vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model4.values,i)for i in range(model4.shape[1])]
vif["feature"]=model4.columns
vif.round(5)
m=smf.OLS(y,model4).fit()
m.summary()

model5=model4

model5=model5.drop("fractal_dimension_worst",axis=1)

#model5 vif and correlation

vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model5.values,i)for i in range(model5.shape[1])]
vif["feature"]=model5.columns
vif.round(6)
m=smf.OLS(y,model5).fit()
m.summary() 


model6=model5

model6=model6.drop("compactness_mean",axis=1)

#model6

vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model6.values,i)for i in range(model6.shape[1])]
vif["feature"]=model6.columns
vif.round(7)
m=smf.OLS(y,model6).fit()
m.summary()


model7=model6

model7=model7.drop("concavity_se",axis=1)

#model7
vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model7.values,i)for i in range(model7.shape[1])]
vif["feature"]=model7.columns
vif.round(8)
m=smf.OLS(y,model7).fit()
m.summary()

model8=model6

model8=model8.drop("concave points_se",axis=1)

#model8
vif=pd.DataFrame()
vif["vif factor"]=[variance_inflation_factor(model8.values,i)for i in range(model8.shape[1])]
vif["feature"]=model8.columns
vif.round(9)
m=smf.OLS(y,model8).fit()
m.summary()



X=model8

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

lm=linear_model.LinearRegression()
model=lm.fit(X_train,y_train)
model.score(X_test,y_test)

prediction=lm.predict(X_test)

prediction=np.where(prediction>0.5,1,0)




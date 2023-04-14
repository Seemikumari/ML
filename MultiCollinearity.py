# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:02:06 2023

@author: SEEMI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('D:\\numpy\\Transformed_Housing_Data2.csv')
data.head()

##SCALING THE DATASET
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
Y = data['Sale_Price']
X = scaler.fit_transform(data.drop(columns=['Sale_Price']))
X = pd.DataFrame(data=X,columns=data.drop(columns=['Sale_Price']).columns)
X.head()


##CHECKING AND REMOVING MULTICOLLINEARITY
X.corr() #we got many corelations


#pair of independent variables with correlation greater than 0.5
k =X.corr()
#use list comprehension
Z=[[str(i),str(j)] for i in k.columns for j in k.columns if (k.loc[i,j]>abs(0.5))&(i!=j)]
Z, len(Z)

#calculating VIF
#importing variance_inflation_factor function from the statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data=X
##calculating VIF for every column
VIF=pd.Series([variance_inflation_factor(vif_data.values,i)for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF

VIF[VIF==VIF.max()].index[0]

def MC_remover(data):
    vif=pd.Series([variance_inflation_factor(data.values,i) for i in range(data.shape[1])],index=data.columns)
    if vif.max()>5:
        print(vif[vif==vif.max()].index[0],'has been removed')
        data=data.drop(columns=[vif[vif==vif.max()].index[0]])
        return data
    else:
        print('No  Multicollinearity present anymore')
        return data
    
for i in range(7):
    vif_data=MC_remover(vif_data)
    
vif_data.head() 


##remaining columns
#calculating VIF for  remaining columns
VIF =pd.Series([variance_inflation_factor(vif_data.values,i) for i in range(vif_data.shape[1])],index=vif_data.columns)
VIF,len(vif_data.columns) 

#Train/Test set

x=vif_data
y=data ['Sale_Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=101)
x_train.shape,x_test.shape,y_train.shape,y_test.shape

##LINEAR REGRESIION
from sklearn.linear_model import LinearRegression
lr=LinearRegression(normalize=True)
lr.fit(x_train,y_train)

lr.coef_

predictions=lr.predict(x_test)
lr.score(x_test,y_test)

#1.Residuals
residuals=predictions - y_test
residual_table=pd.DataFrame({'residuals':residuals,
                             'predictions':predictions})
residual_table=residual_table.sort_values(by='predictions')

z=[i for i in range(int(residual_table['predictions'].max()))]
k = [0 for i in range (int(residual_table['predictions'].max()))]

plt.figure(dpi=130,figsize=(17,7))

plt.scatter(residual_table['predictions'],residual_table['residuals'],color='red',s=2)
plt.plot(z,k,color='green',linewidth=3,label='regression line')
plt.ylim(-800000,800000)
plt.xlabel('fitted points(ordered by predictions)')
plt.ylabel('residuals')
plt.title('residual plot')
plt.legend()
plt.show()

##DISTRIBUTION OF ERRORS
plt.figure(dpi=100,figsize=(10,7))
plt.hist(residual_table['residuals'],color='red',bins=200)
plt.xlabel('residuals')
plt.ylabel('frequency')
plt.title('distribution of residuals')
plt.show()


###MODEL COEFFICIENTS
coefficients_table=pd.DataFrame({'column':x_train.columns,
                                 'coefficients':lr.coef_})
coefficient_table=coefficients_table.sort_values(by='coefficients')

plt.figure(figsize=(8,6),dpi=120)
x=coefficient_table['column']
y=coefficient_table['coefficients']
plt.barh(x,y)
plt.xlabel("Coefficients")
plt.ylabel('variables')
plt.title('Normalized coeffiecient plot')
plt.show()

data.head().shape

data.isnull().sum()
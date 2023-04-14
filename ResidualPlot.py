# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 18:23:03 2023

@author: SEEMI
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Transformed_Housing_Data2.csv')
data.head()
plt.figure(dpi=100)
k=range(0,len(data))
plt.scatter(k,data['Sale_Price'].sort_values(),color='red',label='ACtual Sale Price')
plt.plot(k,data['mean_sales'].sort_values(),color='green',label='mean price')
plt.xlabel('Fitted points(Asdending)')
plt.ylabel("Sale Price")
plt.title('Overall Mean')
plt.legend()

grades_mean=data.pivot_table(values = 'Sale_Price', columns = 'Overall Grade', aggfunc = np.mean)
grades_mean

data['mean_sales']=data['Sale_Price'].mean()
data['mean_sales'].head()

grades_mean=data.pivot_table(values = 'Sale_Price', columns = 'Overall Grade', aggfunc = np.mean)
grades_mean

#making new column
data['grade_mean']=0

#for every grade fill its mean price in new column
for i in grades_mean.columns:
    data['grade_mean'][data['Overall Grade'] == i]=grades_mean[i][0]
    
data['grade_mean'].head() 

# Residual = Prediction - Actual
#A residual plot is a scatter plot of difference between prediction and actual
mean_difference =data['mean_sales']-data['Sale_Price']
grade_mean_difference=data['grade_mean']-data['Sale Price']  
k =range(0,len(data))
l = [0 for i in range(len(data))]#a list of zeros which will represent the residual of a perfect model where predictions are exactly the same as actuals and hence the residuals would be zero

plt.figure(figsize=(15,6),dpi=100)

plt.subplot(1,2,1)
plt.scatter(k,mean_difference,color='red',label='Residuals',s=2)
plt.plot(k,l,color='green',label='mean Regression', linewidth=3)
plt.xlabel('Fitted points')
plt.ylabel("Residuals")
plt.legend()
plt.title("Residuals with respect to Gradewise Mean")

plt.legend() 
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 21:21:45 2023

@author: SEEMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\SEEMI\\.spyder-py3\\Transformed_Housing_Data2.csv')
data.head()
sale_price=data["Sale_Price"].head(30)
flat_area=data['Flat Area (in Sqft)'].head(30)
sample_data=pd.DataFrame({'sale_price':sale_price,'flat_area':flat_area})
sample_data

plt.figure(dpi=150)
plt.scatter(sample_data.flat_area,sample_data.sale_price,color='red')
plt.xlabel('Flat_Area')
plt.ylabel('Sale_Price')
plt.title("Sale_Price/Flat_Area")
plt.legend()
plt.show

sample_data['mean_sale_price']=sample_data.sale_price.mean()

plt.figure(dpi=150)
plt.scatter(sample_data.flat_area,sample_data.sale_price,color='red')
plt.plot(sample_data.flat_area,sample_data.mean_sale_price,color='yellow',label='Mean Sale Price')
plt.xlabel('Flat_Area')
plt.ylabel('Sale_Price')
plt.title("Sale_Price/Flat_Area")
plt.legend()
plt.show

c=0
m=0
line=[]



for i in range(len(sample_data)):
    line.append(sample_data.flat_area[i]*m+c)
    
plt.figure(dpi=130)
plt.scatter(sample_data.flat_area,sample_data.sale_price)
plt.plot(sample_data.flat_area,line,label='m=0.0;c=0')
plt.xlabel('Flat_Area')
plt.ylabel('Sale_Price')
plt.legend()
MSE=mse(sample_data.sale_price,line)    
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:10:33 2023

@author: SEEMI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv('C:\\Users\\SEEMI\\.spyder-py3\\Transformed_Housing_Data2.csv')
data.head()

data['mean_sales']=data['Sale_Price'].mean()
data['mean_sales'].head()

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

#making new column
data['grade_mean']=0

#for every grade fill its mean price in new column
for i in grades_mean.columns:
    data['grade_mean'][data['Overall Grade'] == i]=grades_mean[i][0]
    
data['grade_mean'].head() 

gradewise_list = []
for i in range(1,11):
    k = data["Sale_Price"][data["Overall Grade"] == i]
    gradewise_list.append(k)
    
classwise_list=[]
for i in range(1,11):
    k = data["Sale_Price"][data["Overall Grade"] == i]
    classwise_list.append(k)
    
plt.figure(dpi=120 ,figsize = (15,9))
#plotting sale price grade wise
#z variable is for x axis
z=0
for i in range(1,11):
    #defining x axis using z
    points = [k for k in range(z,z+ len(classwise_list[i-1]))]
    #plotting
    plt.scatter(points,classwise_list[i-1].sort_values(),
                label=('houses with overall grade',i),s=4)
    
#plotting gradewise mean
plt.scatter(points,
            [classwise_list[i-1].mean() for q in range(len(classwise_list[i-1]))],
            s =6,color='pink')
z= max(points)+1

#plotting overall mean
plt.scatter([q for q in range(0,z)],
            data['mean_sales'],
            color='red',
            label ='Overall Mean',
            s=6)

plt.xlabel('Fitted points(Ascending)')
plt.ylabel('Sale Price')
plt.title('Overall Mean')
plt.legend(loc = 4) 

mean_difference=data['mean_sales']-data['Sale_Price']
grade_mean_difference=data['grade_mean']-data['Sale_Price']


    
k =range(0,len(data))
l = [0 for i in range(len(data))]#a list of zeros which will represent the residual of a perfect model where predictions are exactly the same as actuals and hence the residuals would be zero

plt.figure(figsize=(15,6),dpi=100)

plt.subplot(1,2,1)
plt.scatter(k,mean_difference,color='red',label='Residuals',s=2)
plt.plot(k,l,color='green',label='mean Regression', linewidth=3)
plt.xlabel('Fitted points')
plt.ylabel("Residuals")
plt.title("Residuals with respect to Gradewise Mean")
plt.subplot(1,2,2)
plt.scatter(k,grade_mean_difference,color='red',label='Residuls',s=2)
plt.plot(k,l,color='green',label='mean Regression',linewidth=3)
plt.xlabel("Fitted points")
plt.ylabel("Residuals")
plt.legend()
plt.title("Residuals with respect to Gradwise Mean")
plt.legend()


##model evaluation metrices
#1.Mean Absolute Error
#2.Mean Square Error
#3. Root Mean Square Error

cost=sum(mean_difference)/len(data)
print(round(cost,7))

Y= data['Sale_Price']
Y_hat1=data["mean_sales"]
Y_hat2=data['grade_mean']
n= len(data)
len(Y),len(Y_hat1),len(Y_hat2),n

cost_mean=sum(abs(Y_hat1-Y))/n
cost_mean

cost_grade_mean=sum(abs(Y_hat2-Y))/n
cost_grade_mean

#the same thing can be calculated using sklearn library
from sklearn.metrics import mean_absolute_error
cost_grade_mean=mean_absolute_error(Y_hat2,Y)
cost_grade_mean

from sklearn.metrics import mean_squared_error
cost_mean=mean_squared_error(Y_hat1,Y)
cost_grade_mean=mean_squared_error(Y_hat2,Y)
cost_mean,cost_grade_mean

from sklearn.metrics import mean_squared_error
cost_mean=mean_squared_error(Y_hat1,Y)**0.5
cost_grade_mean=mean_squared_error(Y_hat2,Y)**0.5
cost_mean,cost_grade_mean

#R^2 metrics
Y=data["Sale_Price"]
Y_bar=data["mean_sales"]
Y_hat=data['grade_mean']
n=len(data)
len(Y),len(Y_bar),len(Y_hat),n
#now let's calculate mean square error of model 1
mse_mean=mean_squared_error(Y_bar,Y)
mse_mean
#now mean square error of model 2
mse=mean_squared_error(Y_hat,Y)
mse

R2 = 1-(mse/mse_mean)
R2

#LINEAR REGRESIION

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
plt.title("Slope"+str(m)+"with MSE"+str(MSE)) 
  

#this time just change the value of m

c=0
m=50
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
plt.title("Slope"+str(m)+"with MSE"+str(MSE)) 

def slope_Error(slope,intercept,sample_data):
    sale=[]
    for i in range(len(sample_data.flat_area)):
        tmp=sample_data.flat_area[i]*slope+intercept
        sale.append(tmp)
MSE=mse(sample_data.sale_price,sale)
return MSE  

slope=[i/10 for i in range(0,5000)]
Cost=[]
for i in slope:
    cost = slope_Error(slope=i,intercept=10834,sample_data=sample_data)
    Cost.append(cost)
    
#arranging in DataFrame
Cost_table=pd.DataFrame({
    'slope':slope,
    'Cost':Cost})
Cost_table.tail()  

##Plotting the cost values corresponding to every value of beta
plt.plot(Cost_table.slope,Cost_table.Cost,label='Cost Funstion Curve')
plt.xlabel('Value of slope')
plt.ylabel('Cost')
plt.legend()
 

new_slope=Cost_table['slope'][Cost_table['Cost']==Cost_table['Cost'].min()]
new_slope[0]

def intercept_Error(slope,intercept,sample_data):
    sale=[]
    for i in range(len(sample_data.flat_area)):
        tmp=sample_data.flat_area[i]*slope+intercept
        sale.append(tmp)
    MSE =mse(sample_data.sale_price,sale)
    return MSE    

intercept=[i for i in range(5000,50000)]
Cost=[]
for i in intercept:
    cost=intercept_Error(slope=234,intercept=i,sample_data=sample_data)
    Cost.append(cost)
    
#plotting the cost values corresponding to every value of beta  
plt.plot(Cost_table.intercept,Cost_table.Cost,label='Cost Function Curve')
plt.xlabel('Value of intercept')
plt.ylabel('Cost')
plt.legend()  

new_slope=Cost_table['slope'][Cost_table['Cost']==Cost_table['Cost'].min()].values
new_slope[0]






import numpy as np
#STEP 1:INITIALIZE PARAMETER
def param_init(Y):
    '''
    

    Parameters
    ----------
    Y : target variable returns initialized values
    of m and c.

    '''
    m=0.1
    c=Y.mean()
    return m,c

#STEP 2:GENERATE PREDICTIONS

def generate_predictions(m, c, X):
    '''
    

    Parameters
    ----------
    m : slope.
    c : intercept.
    X : independent variable returns prediction
    generated by line with parameter m,c.

    Returns
    it returns a list of prediction

    '''
    
    prediction =[]
    for x in X:
        pred=(m *x )+c
        prediction.append(pred)
    return prediction    
    
    
##STEP 3: CALCULATING COST   
def compute_cost(prediction,Y):
    '''
    returns the mean_squared _error between prediction and Y

    Parameters
    ----------
    prediction : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    cost = np.sum(((prediction-Y)**2)/len(Y))
    return cost

##STEP 4: UPDATING PARAMETERS
def gradients(prediction ,Y,X):
    '''
    returns gradients corresponding to m and c

    Parameters
    ----------
    prediction : TYPE
        DESCRIPTION.
    Y : TYPE
        DESCRIPTION.
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    n = len(Y)
    Gm = 2/n * np.sum((prediction-Y)*X)
    Gc =2/n * np.sum((prediction-Y))
    return Gm ,Gc

def param_update(m_old,c_old,Gm_old,Gc_old,alpha):
    '''
    update and return the new values of m and c
    '''
    m_new = m_old-alpha*Gm_old
    c_new=c_old-alpha*Gc_old
    return m_new,c_new


def result(m,c,Y,X,cost,predictions,i):
    '''
    Print and plot the final result obtained from gradient descent
    '''
    ##if the gradient descent converged to the optimum value before max_iter
    if x < max_iter -1:
        print("***** Gradient Descent has converged at iteration{}*****".format(i))
    else:
        print("*****Result after", max_iter,'itertions is:  *****')
        
        
##plotting the final result
plt.figure(figsize=(14,7),dpi=120)
plt.scatter(X,Y,color='red',label='data points')
label='final regression line: m ={}; c = {}'.format(str(m).str(c))
plt.plot(X,predictions,color='green',label=label)
plt.xlabel('flat_area')
plt.ylabel('sale_price')
plt.title('final regression line')
plt.legend()        
        
        


##defining and reshaping the dataset
sale_price=sample_data['sale_price'].value.reshape(-1,1)
flat_area=sample_data['flat_area'].values.reshape(-1,1)
##scaling the dataset using standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
sale_price=scaler.fit_transform(sample_data['sale_price'].value.reshape(-1,1))
flat_area=scaler.fit_transform(sample_data['flat_area'].values.reshape(-1,1))
##decalring parameters
max_iter=1000
cost_old =0
alpha=0.01

##step 1:intializing the values of m,c
m,c = param_init(sale_price)
 ##gradient Descent in action
for i in range(0, max_iter):
     
##step 2: generating predictions
   predictions=generate_predictions(m,c,flat_area)
##step 3: calculating cost  
cost_new=compute_cost(predictions,sale_price)
##checking if GD converged
if abs(cost_new - cost_old) < 10**(-7):
    break
  

##calculating gradients
Gm,Gc = gradients(predictions,sale_price,flat_area)

##step 4:updating parameters m and c
m,c=param_update(m,c,Gm,Gc,alpha)

##display result after every 20 iterations
if i%20 == 0:
    print('After Iteration', i,': m =', m, '; c =', c,'; Cost =', cost_new)

  ##updating cost_old
cost_old=cost_new
##final results
result(m,c,flat_area,sale_price,
       cost_new,predictions,i)    

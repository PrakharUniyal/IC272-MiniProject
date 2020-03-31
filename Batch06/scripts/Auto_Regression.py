# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:54:57 2019
@author: prakhar
"""

#2-Aug-2018 to 31-Dec-2018
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#read data from csv file
data = pd.read_csv("group6.csv")


#converting string time to numerical time in units of minutes
time=data.iloc[:,0]
#time_mins=pd.Series(np.arange(0,185775,15))


#updating data df with the converted time
#data.iloc[:,0]=time_mins
data=data.iloc[:,1:]
col_names=data.columns


#splitting data to train and test data
l=len(data)
index=int(l*0.7)
train,test=data.iloc[:index,:],data.iloc[index:,:]

#normalisation
from sklearn.preprocessing import MinMaxScaler
norm_model=MinMaxScaler()
norm_train_data=norm_model.fit_transform(train)
norm_test_data=norm_model.transform(test)

#standardisation
from sklearn.preprocessing import StandardScaler
standard_model=StandardScaler()
std_train_data=standard_model.fit_transform(train)
std_test_data=standard_model.transform(test)


time_arr=time.values
train_time=time_arr[:index]
test_time=time_arr[index:]
'''
Line plots for attributes wrt time on training data
'''
for i in range(len(col_names)):
    f=plt.figure(i+1)
    plt.plot(train_time,train.iloc[:,i],linewidth=1)
    plt.ylabel(col_names[i])
    plt.xlabel('Time')
    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 18})
    #plt.savefig(col_names[i]+'_time'+'.png',dpi=300)
    f.show()


'''
AUTO REGRESSION
'''
import statsmodels.graphics.tsaplots as st
from statsmodels.tsa.ar_model import AR

#predictive analysis on NORM
train_data=norm_train_data
test_data=norm_test_data
for i in range(train_data.shape[1]):
    st.plot_acf(train_data[:,i],lags=37,vlines_kwargs={'linewidth':5})
    plt.xlabel('Time Lag')
    plt.ylabel('Pearson Coefficient')
    plt.title(col_names[i]+' (Normalised Data)')
    matplotlib.rcParams.update({'font.size': 18})
    plt.tight_layout()
    #plt.savefig(col_names[i]+'_norm'+'.png')
    plt.show()

optiml_lag_norm=[]
rmse_error_non_dynamic_norm=[]
rmse_error_dynamic_norm=[]
for i in range(train_data.shape[1]):
    #print(col_names[i])
    model = AR(train_data[:,i])
    '''
    If ic is not specified --> default :
    statsmodels uses default round(12*(nobs/100.)**(1/4.)) to determine the number of lags to use.
    
    If ic = t-stat, the model starts with maxlag and drops a lag until the highest lag has a t-stat that is
    significant at the 95 % level.
    
     bic - Bayes Information Criterion
    
    '''
    model_fitted = model.fit(ic='bic')
    
    optiml_lag_norm.append(model_fitted.k_ar)
#    if(i==4):
#        print(col_names[i])
#        print(model_fitted.params)
    
    predictions_non_dynamic = model_fitted.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
    predictions_dynamic = model_fitted.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=True)
    
    rmse_non_dynamic=mean_squared_error(test_data[:,i], predictions_non_dynamic)**0.5
    rmse_error_non_dynamic_norm.append(rmse_non_dynamic)
    
    rmse_dynamic=mean_squared_error(test_data[:,i], predictions_dynamic)**0.5
    rmse_error_dynamic_norm.append(rmse_dynamic)
        
    

#predictive analysis on STD
train_data=std_train_data
test_data=std_test_data
for i in range(train_data.shape[1]):
    st.plot_acf(train_data[:,i],lags=37)
    plt.xlabel('Time Lag')
    plt.ylabel('Pearson Coefficient')
    plt.title(col_names[i]+' (Standardised Data)')
    plt.show()

optiml_lag_std=[]
rmse_error_non_dynamic_std=[]
rmse_error_dynamic_std=[]
for i in range(train_data.shape[1]):
    #print(col_names[i])
    model = AR(train_data[:,i])
    '''
    If ic is not specified --> default :
    statsmodels uses default round(12*(nobs/100.)**(1/4.)) to determine the number of lags to use.
    
    If ic = t-stat, the model starts with maxlag and drops a lag until the highest lag has a t-stat that is
    significant at the 95 % level.
    
     bic - Bayes Information Criterion
    
    '''
    model_fitted = model.fit(ic='bic')
    
    optiml_lag_std.append(model_fitted.k_ar)
#    if(i==4):
#        print(col_names[i])
#        print(model_fitted.params)
    
    predictions_non_dynamic = model_fitted.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
    predictions_dynamic = model_fitted.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=True)
    
    rmse_non_dynamic=mean_squared_error(test_data[:,i], predictions_non_dynamic)**0.5
    rmse_error_non_dynamic_std.append(rmse_non_dynamic)
    
    rmse_dynamic=mean_squared_error(test_data[:,i], predictions_dynamic)**0.5
    rmse_error_dynamic_std.append(rmse_dynamic)
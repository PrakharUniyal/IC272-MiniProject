# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import datetime
cmap=sns.diverging_palette(5, 250, as_cmap=True)

<<<<<<< HEAD
def unix(d): return int(time.mktime(datetime.datetime.strptime(d, "%Y-%m-%d %H:%M:%S").timetuple()))

def datestosrno(datecol):    
    dates = datecol.apply(unix)
    start = dates.iloc[0]
    intval = dates.iloc[1]-dates.iloc[0]
    return dates.apply(lambda t : (t-start)/intval)

def replace_outliers(df):
    #Replaces outliers(according to boxplot) with median value in each attribute.
    for i in list(df):
        q1,q2,q3  = df[i].quantile(0.25),df[i].quantile(0.50),df[i].quantile(0.75)
        lb,ub = (q1 - (3/2)*(q3-q1)) , (q3 + (3/2)*(q3-q1))
        print("Change in no. of outliers in",i,":",df[i][df[i]<lb].count()+df[i][df[i]>ub].count(),"--> ",end='')
        
        df[i].mask(df[i]<lb,q2,inplace=True)
        df[i].mask(df[i]>ub,q2,inplace=True)
        
        q1,q2,q3  = df[i].quantile(0.25),df[i].quantile(0.50),df[i].quantile(0.75)
        lb,ub = (q1 - (3/2)*(q3-q1)) , (q3 + (3/2)*(q3-q1))
        print(df[i][df[i]<lb].count()+df[i][df[i]>ub].count())
    return df

data = pd.read_csv("group6.csv")

'''Getting all the column names.'''
print(list(data))
'''"MemoryUsed"(response variable attribute) is the 5th column in dataframe.'''


'''Checking whether there is any missing value in any column.'''
#print(data.isna().sum())
'''No Missing Values Found'''


'''Plotting the histograms for each attribute.'''
#pd.DataFrame.hist(data,figsize=(15,15))
'''Saved.'''

'''Plotting the boxplots'''
#for i in list(data.iloc[:,1:]): plt.boxplot(data[i]); plt.xlabel(i); plt.show()
'''Saved the plots'''

'''Getting the Correlation data for all attributes and doing descriptive analysis'''
#corr = data.corr()
#plt.figure(figsize=(12,8))
#plt.subplot(111)
#sns.heatmap(corr,annot=True)
#plt.show()
#data.describe().to_csv("DescriptiveAnalysis.csv")
'''Files Created'''


'''Checking whether there is any attribute with non-numerical data.'''
#print([data.iloc[:,i].dtype for i in range(len(list(data)))])
'''All the attributes except the "CreationTime"(timestamp) have dtype float64.'''

dates = datestosrno(data.iloc[:,0])
ndata = pd.concat([datestosrno(data.iloc[:,0]),replace_outliers(data.iloc[:,1:].copy())],axis=1)

'''Scatter plots for each attribute'''
#for i in range(1,len(list(ndata))):
#    plt.scatter(ndata.iloc[:,0],ndata.iloc[:,i],marker='.')
#    plt.xlabel(list(data)[i])
#    plt.show()
'''Saved the files'''

##################################################################################3
#Split the response variable attribute from predictor variable attributes.
#X = pd.concat([ndata.iloc[:,1:5],ndata.iloc[:,6:]],axis=1)
#Y = ndata.iloc[:,5]
X = pd.concat([data.iloc[:,1:5],data.iloc[:,6:]],axis=1)
Y = data.iloc[:,5]


from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
mX = minmax.fit_transform(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sX = scaler.fit_transform(X)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
def rmse(y_test,y_pred): return mse(y_test,y_pred)**(0.5)
from sklearn.model_selection import train_test_split

def Linear_reg(X,Y):
    x_train, x_test, y_train, y_test =  train_test_split(X,Y,test_size = 0.3, random_state= 42)
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    print("RMSE train:",rmse(y_train,y_train_pred))
    print("RMSE test :",rmse(y_test,y_test_pred))
    print("Model Co-eff and intercept:",model.coef_,model.intercept_)
    print("R2 score train:",r2_score(y_train,y_train_pred))
    print("R2 score test :",r2_score(y_test,y_test_pred))
    plt.scatter(y_train,y_train_pred); plt.show()
    plt.scatter(y_test,y_test_pred); plt.show()

Linear_reg(pd.DataFrame(X["MemoryFree"]),Y)
print()
Linear_reg(X,Y)
Linear_reg(mX,Y)
Linear_reg(sX,Y)
    
"""Reducing the data using pca"""
from sklearn.decomposition import PCA

def pca_red(X,Y):
    train_error = []
    test_error = []
    
    for i in range(1,1+len(list(X))):
        pca = PCA(n_components = i)
        tX = pca.fit_transform(X)
        x_train, x_test, y_train, y_test =  train_test_split(tX,Y,test_size = 0.3, random_state= 42)    
        model = LinearRegression()
        model.fit(x_train,y_train)  
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        train_error.append(rmse(y_train,y_train_pred))
        test_error.append(rmse(y_test,y_test_pred))
        
    return (train_error,test_error)

print(pca_red(X,Y)[0])
print(pca_red(X,Y)[1])

"""Best dimension found = 10"""

pca = PCA(n_components=10)
tenX = pca.fit_transform(X)

pca = PCA(n_components=5)
fiveX = pca.fit_transform(X)

Linear_reg(tenX,Y)

################################################################################
from sklearn.preprocessing import PolynomialFeatures

def multivar_poly(X,Y):
     x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
     p = [2,3,4]
     
     r_train = []
     r_test = []
     for i in p:
         polynomial_features = PolynomialFeatures(i)
         x_poly = polynomial_features.fit_transform(x_train)
         regressor = LinearRegression()
         regressor.fit(x_poly, y_train)
         y1 = regressor.predict(x_poly)
         print("For degree of polynomial = ", i)
         print("Prediction accuracy for training data = ", rmse(y1,y_train))
         r_train.append(rmse(y1, y_train))
         x1 = polynomial_features.fit_transform(x_test)
         y_pred = regressor.predict(x1)
         print("prediction accuracy for test data = ", rmse(y_pred,y_test))
         r_test.append(rmse(y_pred, y_test))

         print(regressor.coef_,regressor.intercept_)

         plt.scatter(y_train, y1)
         plt.xlabel("actual train data")
         plt.ylabel("predicted train data")
         plt.show()             
         
         plt.scatter(y_test, y_pred)
         plt.xlabel("actual test data")
         plt.ylabel("predicted test data")
         plt.show()
         
     bars_train = plt.bar(p, r_train, width = .3)
     for bar in bars_train:
         yval = round(bar.get_height(),5)
         plt.text(bar.get_x(), yval + .005, yval)
     plt.xlabel("degree of polynomial")
     plt.ylabel("prediction accuracy")
     plt.title("training data")
     plt.show()
     
     bars_test = plt.bar(p, r_test, width = .3)
     for bar in bars_test:
         yval = round(bar.get_height(),5)
         plt.text(bar.get_x(), yval + .005, yval)
     plt.xlabel("degree of polynomial")
     plt.ylabel("prediction accuracy")
     plt.title("test data")
     plt.show()
     
#multivar_poly(pd.DataFrame(X["MemoryFree"]),Y)
print()
#multivar_poly(X,Y)
#multivar_poly(mX,Y)
#multivar_poly(sX,Y)
multivar_poly(fiveX,Y)
=======
data = pd.read_csv("group6.csv")

#Getting all the column names.
#print(list(data))
'''"MemoryUsed"(response variable attribute) is the 5th column in dataframe.'''

#Split the response variable attribute from predictor variable attributes.
X = pd.concat([data.iloc[:,:5],data.iloc[:,6:]],axis=1)
Y = pd.DataFrame(data.iloc[:,5])

#Checking whether there is any missing value in any column.
#print(data.isna().sum())
'''No Missing Values Found'''

#Checking whether there is any attribute with non-numerical data.
#print([data.iloc[:,i].dtype for i in range(len(list(data)))])
'''All the attributes except the "CreationTime"(timestamp) have dtype float64.'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =  train_test_split(X,Y,test_size = 0.3, random_state= 42)

#data.corr().to_csv("Correlation.csv")

>>>>>>> 5242dcd1fae9ae8ba02b3cf9250b7549638dcc2b

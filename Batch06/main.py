# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


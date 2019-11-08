# -*- coding: utf-8 -*-
#b18115 - Harish Jaglan
#b18037 - Vivek Moar
import pandas as pd

data = pd.read_csv("group6.csv")

#Checking whether there is any missing value in any column.
print(data.isna().sum())
'''No Missing Values Found'''

# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 17:37:44 2020

@author: black
age - Age of Customer
ed - Eductation level of customer
employ: Tenure with current employer (in years)
address: Number of years in same address
income: Customer Income
debtinc: Debt to income ratio
creddebt: Credit to Debt ratio
othdebt: Other debts
default: Customer defaulted in the past (1= defaulted, 0=Never defaulted) 
"""
import pandas as pd
import zipfile
#import numpy as np 
import seaborn as sns
from pandas import DataFrame as DF
import matplotlib.pyplot as plt

zf = zipfile.ZipFile('bankloans.zip') 
loans = pd.read_csv(zf.open('bankloans.csv'))

#EDA BEGINS BELOW-----------------    
print(loans.isna().sum())  #missing only some labels
unlabeled = loans[loans.isna().any(axis=1)].reset_index(drop=True).drop(
        columns=['default'])  #saving unlabeled data for unsupervised?
loans.dropna(inplace=True) #missing labels, how else to handle?

#Visualizing label distribution
plt.title('Defaults (1) vs Never Defaulted (0)')
plt.xlabel('Default category')
plt.ylabel('Sampled Individuals')
plt.grid(True)
sns.barplot(data=DF(loans.default.value_counts()).T)

unlabeled = loans[loans.isna().any(axis=1)].reset_index(drop=True).drop(
        columns=['default'])  #saving unlabeled data for unsupervised?
loans.dropna(inplace=True) #missing labels, how else to handle?

#Descriptive statistics
pd.set_option('display.expand_frame_repr', False)
print(loans.describe().T)


#Correlations
corrs = loans.corr()
sns.heatmap(corrs, annot=True)

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


# (Super basic) EDA:
sns.set_style('darkgrid'); sns.set_palette('muted')

sns.boxplot(x='ed', y='employ', data=loans, hue='default')
plt.show()

sns.boxplot(x='ed', y='address', data=loans, hue='default')
plt.show()

sns.jointplot("debtinc", "default", data=loans, kind="kde")
plt.show()

sns.jointplot("creddebt", "default", data=loans, kind="kde")
plt.show()

sns.violinplot(x='ed', y='income', hue='default', data=loans)
plt.show()

sns.scatterplot(x='debtinc', y='othdebt', hue='default', data=loans)
plt.show()

sns.scatterplot(x='age', y='income' , hue='default', data=loans)
plt.show()

#Descriptive stats based on default, quite informative
group = loans.groupby('default').median()
print(group)

#tsne (PCA for models maybe not useful, small dataset anyway)

from sklearn.manifold import TSNE 
tsne = TSNE(random_state=seed)
loansts = loans.copy()
tsne_features = tsne.fit_transform(loansts)
loansts['x'], loansts['y'] = tsne_features[:, 0], tsne_features[:,1]
sns.scatterplot(x='x', y='y', hue='default', data=loansts)
plt.show()

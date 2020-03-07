#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

#Load the training data
df_train=pd.read_csv('train.csv')

#Check the column headers
print(df_train.columns)

#Descriptive statistics summary
df_train['SalePrice'].describe()

#histogram
sns.distplot(df_train['SalePrice'])

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

#check if this variable is not relevant to the sale price
var = 'EnclosedPorch'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#check duplicated rows
df_train[df_train.duplicated()==True]

#check column data types
res = df_train.dtypes
print(res[res == np.dtype('int64')])
print(res[res == np.dtype('bool')])
print(res[res == np.dtype('object')])
print(res[res == np.dtype('float64')])

#standardize
print(df_train["LotConfig"].unique())

# feature scaling, only apply to numeric data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(df_train[["GrLivArea","SalePrice"]])
sns.distplot(X_train[:,1],fit=norm)

#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm)
# fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'], plot=plt)

#data transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Percent'] > 0.15]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

#if not drop, we can impute
# check correlation with LotArea
df_train['LotFrontage'].corr(df_train['LotArea'])

df_train['SqrtLotArea']=np.sqrt(df_train['LotArea'])
df_train['LotFrontage'].corr(df_train['SqrtLotArea'])

cond = df_train['LotFrontage'].isnull()
df_train["LotFrontage"][cond]=df_train["SqrtLotArea"][cond]
print(df_train["LotFrontage"].isnull().sum())

#flag the missing data as missing
mis=df_train['GarageType'].isnull()
df_train["GarageType"][mis]="Missing"
df_train["GarageType"].unique()

#identify the outliers
fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(16, 4))
axes = np.ravel(axes)
col_name = ['GrLivArea','TotalBsmtSF','1stFlrSF','BsmtFinSF1','LotArea']
for i, c in zip(range(5), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='r')

#delete outliers
print(df_train.shape)
df_train = df_train[df_train['GrLivArea'] < 4500]
df_train = df_train[df_train['LotArea'] < 100000]
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
df_train = df_train[df_train['1stFlrSF'] < 2500]
df_train = df_train[df_train['BsmtFinSF1'] < 2000]

print(df_train.shape)

for i, c in zip(range(5,10), col_name):
    df_train.plot.scatter(ax=axes[i], x=c, y='SalePrice', sharey=True, colorbar=False, c='b')
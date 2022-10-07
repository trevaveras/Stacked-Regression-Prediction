
# Feature Engineering Script for Ames Housing Dataset

# Credit where it is due: https://www.kaggle.com/code/lucabasa/houseprice-end-to-end-project/notebook

def ftrengineer(train,test):

    import os
    import numpy as np
    import pandas as pd
    import re
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    from scipy import stats


    def neighbord(col):
        if col in ['Edwards','BrkSide',
                   'OldTown','BrDale',
                   'IDOTRR','MeadowV']:
            return 0
        elif col in ['Mitchel','NAmes',
                     'NPkVill','SWISU',
                     'Blueste','Sawyer'] :
            return 1
        elif col in ['Crawfor','CollgCr',
                     'Blmngtn','Gilbert',
                     'NWAmes','SawyerW']:
            return 2
        elif col in ['NoRidge','NridgHt',
                     'StoneBr','Timber',
                     'Veenker','Somerst','ClearCr']:
            return 3


    train['Neighborhood']=train['Neighborhood'].apply(neighbord)
    test['Neighborhood']=test['Neighborhood'].apply(neighbord)

    def ordinalqual(col):
        if col=='NA':
            return 0
        if col=='Po':
            return 1
        elif col=='Fa':
            return 2
        elif col=='TA':
            return 3
        elif col=='Gd':
            return 4
        elif col=='Ex':
            return 5


    # ordinal encode these quality and condition features
    for i in ['ExterQual','ExterCond',
              'BsmtQual','BsmtCond',
              'KitchenQual','FireplaceQu',
              'GarageCond']:
        train[i]=train[i].apply(ordinalqual)

    # ordinal encode these quality and condition features
    for i in ['ExterQual','ExterCond',
              'BsmtQual','BsmtCond',
              'KitchenQual','FireplaceQu',
              'GarageCond']:
        test[i]=test[i].apply(ordinalqual)

    # lot shape reg vs other
    def lotter(col):
        if col=='Reg':
            return 'Reg'
        elif col!='Reg':
            return 'Irregular'

    train['LotShape']=train['LotShape'].apply(lotter)
    test['LotShape']=test['LotShape'].apply(lotter)



    def logger(data,col):
        data[col]=np.log(data[col])
        return data

    train=logger(train, 'GrLivArea')
    test=logger(test, 'GrLivArea')
    train=logger(train, '1stFlrSF')
    test=logger(test, '1stFlrSF')
    train=logger(train, 'LotArea')
    test=logger(test, 'LotArea')


    def garager(col):
        if col=='None':
            return 'NA'
        elif col=='NA':
            return 'NA'
        elif col=='CarPort':
            return 'NA'
        elif col=='Basment':
            return 'NA'
        elif col=='2Types':
            return 'NA'
        elif col=='BuiltIn':
            return 'BuiltIn'
        elif col=='Attchd':
            return 'Attchd'
        elif col=='Detchd':
            return 'Detchd'
    train['GarageType']=train['GarageType'].apply(garager)
    test['GarageType']=test['GarageType'].apply(garager)

    # define a few new features

    def SF_per_room(data):
        data['sf_per_room'] = data['GrLivArea'] / data['TotRmsAbvGrd']
        return data

    def bedroom_prop(data):
        data['bedroom_prop'] = data['BedroomAbvGr'] / data['TotRmsAbvGrd']
        return data

    train = SF_per_room(train)
    train = bedroom_prop(train)
    test = SF_per_room(test)
    test = bedroom_prop(test)


    def total_bath(data):
        data['total_bath'] = data[
            [col for col in data.columns if 'FullBath' in col]].sum(
            axis=1)+ 0.5 * data[
            [col for col in data.columns if 'HalfBath' in col]].sum(
            axis=1)
        return data

    train = total_bath(train)
    test = total_bath(test)


    train=train.drop(columns=(['BsmtFullBath','BsmtHalfBath',
                               'FullBath','HalfBath']))
    test=test.drop(columns=(['BsmtFullBath','BsmtHalfBath',
                               'FullBath','HalfBath']))

    return(train, test)
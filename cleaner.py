
# cleaning script for Ames Housing Dataset

def cleanmydata():

    import os
    import numpy as np
    import pandas as pd
    import re
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    from scipy import stats

    os.chdir('./Kaggle')

    train=pd.read_csv('./train.csv')
    test=pd.read_csv('./test.csv')
    
    # TRAINING DATA

    # remove ID
    train=train.drop(columns=('Id'))

    # cast MSSubClass as categorical
    train['MSSubClass']=train['MSSubClass'].astype(object)

    nums = [col for col in train.columns if\
            train[col].dtype=='int64' or train[col].dtype=='float64']

    # missing data imputation on training set
    train['LotFrontage'].fillna(value=train['LotFrontage'].median(), inplace=True)
    train['GarageYrBlt'].fillna(value=train['GarageYrBlt'].median(), inplace=True)
    # Likely to be 'None' type with zero square footage vnr area
    train['MasVnrArea'].fillna(0, inplace=True)

    # save categoricals
    cats = [col for col in train.columns if train[col].dtype=='object']

    # drop list
    train=train.drop(columns=(['MiscFeature', 'PoolQC','Alley']))

    # index 948 is no exposure basement 'No'
    train.at[948,'BsmtExposure']='No'
    # overwrite one missing cell for bsmtfintype2
    train.at[332,'BsmtFinType2']='GLQ'

    # impute missing values
    train['GarageFinish'].fillna('None', inplace=True)
    train['GarageQual'].fillna('None', inplace=True)
    train['GarageCond'].fillna('NA', inplace=True)
    train['GarageType'].fillna('NA', inplace=True)
    train['FireplaceQu'].fillna('NA', inplace=True)
    train['MasVnrType'].fillna('None', inplace=True)
    train['Electrical'].fillna('SBrkr', inplace=True)
    train['BsmtExposure'].fillna('NA', inplace=True)
    train['BsmtFinType1'].fillna('NA', inplace=True)
    train['BsmtFinType2'].fillna('NA', inplace=True)
    train['BsmtQual'].fillna('NA', inplace=True)
    train['BsmtCond'].fillna('NA', inplace=True)
    train['Fence'].fillna('NA', inplace=True)

    # remove outliers for important variables
    train['SalePrice']=train['SalePrice'][(np.abs(stats.zscore(train['SalePrice'])) < 3)]
    train['GrLivArea']=train['GrLivArea'][(np.abs(stats.zscore(train['GrLivArea'])) < 3)]

    # drop rows with NaN and reset index
    train=train.dropna(axis=0).reset_index(drop=True)

    # TEST DATA

    # drop ID from test
    test=test.drop(columns=('Id'))

    # cast MSSubClass as categorical
    test['MSSubClass']=test['MSSubClass'].astype(object)

    nums = [col for col in test.columns if\
            test[col].dtype=='int64' or test[col].dtype=='float64']

    # imputation on test data
    test['LotFrontage'].fillna(value=test['LotFrontage'].median(), inplace=True)
    # Likely to be 'None' type with zero square footage vnr area
    test['MasVnrArea'].fillna(0, inplace=True)

    test['BsmtFinSF1'].fillna(0, inplace=True)
    test['BsmtFinSF2'].fillna(0, inplace=True)
    test['BsmtUnfSF'].fillna(0, inplace=True)
    test['TotalBsmtSF'].fillna(0, inplace=True)

    #garage
    test['GarageYrBlt'].fillna(value=test['GarageYrBlt'].median(), inplace=True)
    test['GarageCars'].fillna(value=test['GarageCars'].median(), inplace=True)  
    test['GarageArea'].fillna(value=test['GarageCars'].median(), inplace=True) 

    test['BsmtFullBath'].fillna(0, inplace=True)
    test['BsmtHalfBath'].fillna(0, inplace=True)

    # save categoricals
    cats = [
        col for col in test.columns if test[col].dtype=='object']

    # drop list
    test=test.drop(columns=(['MiscFeature','PoolQC','Alley']))

    # impute
    test['MSZoning'].fillna(
        value=test['MSZoning'].mode()[0], inplace=True)
    test['Utilities'].fillna(
        value=test['Utilities'].mode()[0], inplace=True)
    test['Exterior1st'].fillna(
        value=test['Exterior1st'].mode()[0], inplace=True)
    test['Exterior2nd'].fillna(
        value=test['Exterior2nd'].mode()[0], inplace=True)

    test['MasVnrType'].fillna(
        value=test['MasVnrType'].mode()[0], inplace=True)

    test['KitchenQual'].fillna(
        value=test['KitchenQual'].mode()[0], inplace=True)


    # overwrite one missing cell for bsmtqual
    test.at[580,'BsmtCond']='Gd'
    test.at[725,'BsmtCond']='Po'
    test.at[1064,'BsmtCond']='Gd'
    test.at[757,'BsmtQual']='Fa'
    test.at[758,'BsmtQual']='TA'
    test.at[27,'BsmtExposure']='Av'
    test.at[888,'BsmtExposure']='Av'

    ## missing or none or zero
    test['BsmtQual'].fillna('NA', inplace=True)
    test['BsmtCond'].fillna('NA', inplace=True)
    test['BsmtExposure'].fillna('NA', inplace=True)       
    test['BsmtFinType1'].fillna('NA', inplace=True)   
    test['BsmtFinType2'].fillna('NA', inplace=True)   

    test['FireplaceQu'].fillna('NA', inplace=True) 

    test['Functional'].fillna(value=test['Functional'].mode()[0], inplace=True)  
    test['SaleType'].fillna(value=test['SaleType'].mode()[0], inplace=True)  

    # impute 666 garage

    test.at[666,'GarageYrBlt']=test['GarageYrBlt'].median()
    test.at[666,'GarageFinish']=test['GarageFinish'].mode()[0]
    test.at[666,'GarageQual']=test['GarageQual'].mode()[0]
    test.at[666,'GarageCond']=test['GarageCond'].mode()[0]

    #impute 1116
    test.at[1116,'GarageYrBlt']=test['GarageYrBlt'].median()
    test.at[1116,'GarageFinish']=test['GarageFinish'].mode()[0]
    test.at[1116,'GarageQual']=test['GarageQual'].mode()[0]
    test.at[1116,'GarageCond']=test['GarageCond'].mode()[0]
    # THIS IS NUMERIC
    test.at[1116,'GarageArea']=test['GarageArea'].median()
    # fill fence
    test['Fence'].fillna('NA', inplace=True)


    test['GarageType'].fillna('NA', inplace=True)
    test['GarageFinish'].fillna('NA', inplace=True)
    test['GarageQual'].fillna('NA', inplace=True)
    test['GarageCond'].fillna('NA', inplace=True)

    return(train, test)



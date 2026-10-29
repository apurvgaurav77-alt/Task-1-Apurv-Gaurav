import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera as pa 
from pandera import Column, DataFrameSchema, Check
df = pd.read_csv('/Users/apurvgaurav/Downloads/week 1.csv')
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
numerical_cols = df.select_dtypes(include=["number"]).columns
print(numerical_cols)
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)
print(df)
print(df.isnull().sum()/len(df)*100)
imputer = KNNImputer(n_neighbors=5)
df[[col]] = imputer.fit_transform(df[[col]])
print(df.isnull().sum())
df['CouponCode'] = df['CouponCode'].fillna('Not applicable')





df = pd.get_dummies(df, columns=['PaymentMethod', 'OrderStatus', 'ShippingAddress', 'ReferralSource'])
corr_matrix = df.corr(numeric_only=True)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
df = df.drop(columns=to_drop)
schema = DataFrameSchema({'ItemsInCart': Column(pa.Int), 'TotalPrice': Column(pa.Float)})
schema.validate(df)
df.to_csv('/Users/apurvgaurav/Downloads/week 1_cleaned.csv', index=False)

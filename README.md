# week-1-decodelabs-project-


# Data Cleaning Project: (Week 1)

## Project Overview

This project cleans and prepares a raw e-commerce orders dataset (`week 1.csv`) for analysis and machine learning. The raw data has **1,200 rows and 14 columns**. After cleaning and encoding, the output (`cleaned_data.csv`) has **1,200 rows and 680 columns**, because categorical columns were one-hot encoded.

All cleaning is done in Python in `data cleaning.py` using pandas, NumPy, scikit-learn and Pandera.

## Files in this Repository

| File | Description |
|------|-------------|
| `data cleaning.py` | Python script that performs the full cleaning pipeline |
| `week 1.csv` | Raw input dataset |
| `cleaned_data.csv` | Cleaned and encoded output dataset |
| `README.md` | Project documentation |

## Dataset Description (Raw Data)

| Column | Type | Description |
|--------|------|-------------|
| OrderID | String | Unique order identifier |
| Date | Date | Order date |
| CustomerID | String | Customer identifier |
| Product | String | Product ordered |
| Quantity | Integer | Units ordered |
| UnitPrice | Float | Price per unit |
| ShippingAddress | String | Delivery address |
| PaymentMethod | String | Cash, Credit Card, Debit Card, Gift Card, Online |
| OrderStatus | String | Cancelled, Delivered, Pending, Returned, Shipped |
| TrackingNumber | String | Shipment tracking ID |
| ItemsInCart | Integer | Number of items in the cart |
| CouponCode | String | Coupon used (if any) |
| ReferralSource | String | Email, Facebook, Google, Instagram, Referral |
| TotalPrice | Float | Total order price |

## Tasks Performed

1. Loaded the raw dataset and inspected it (head, missing values, duplicates, summary statistics)
2. Identified the numerical columns
3. Handled outliers using the IQR method (capping/winsorizing)
4. Checked the percentage of missing values
5. Imputed missing numerical values using KNN Imputer
6. Filled missing categorical values in `CouponCode`
7. One-hot encoded categorical columns
8. Removed highly correlated features (correlation > 0.8)
9. Validated the data schema using Pandera
10. Exported the cleaned dataset to CSV

## Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera as pa
from pandera import Column, DataFrameSchema, Check
```

| Library | Purpose |
|---------|---------|
| `pandas` | Loading, cleaning and transforming tabular data |
| `numpy` | Numerical operations (used to build the upper-triangle mask) |
| `KNNImputer` (scikit-learn) | Fills missing numeric values using the nearest neighbours |
| `pandera` | Validates that the DataFrame matches an expected schema |

## Line-by-Line Code Explanation

### 1. Importing libraries
```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import pandera as pa
from pandera import Column, DataFrameSchema, Check
```
- Imports pandas and NumPy for data handling and math.
- Imports `KNNImputer` to fill missing values.
- Imports Pandera (`pa`) and its `Column`, `DataFrameSchema` and `Check` classes for data validation.

### 2. Loading the data
```python
df = pd.read_csv('/Users/apurvgaurav/Downloads/week 1.csv')
```
- Reads the raw CSV file into a DataFrame called `df`.

### 3. Initial data inspection
```python
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
```
- `df.head()` shows the first 5 rows to get a feel for the data.
- `df.isnull().sum()` counts missing values in each column.
- `df.duplicated().sum()` counts fully duplicate rows.
- `df.describe()` gives summary statistics (count, mean, std, min, quartiles, max) for numeric columns.

### 4. Selecting numerical columns
```python
numerical_cols = df.select_dtypes(include=["number"]).columns
print(numerical_cols)
```
- Selects only the numeric columns (e.g., `Quantity`, `UnitPrice`, `ItemsInCart`, `TotalPrice`) and prints their names.

### 5. Outlier handling with the IQR method
```python
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)
print(df)
```
- Loops over every numeric column.
- `Q1` and `Q3` are the 25th and 75th percentiles.
- `IQR = Q3 - Q1` is the interquartile range.
- `lower_bound` and `upper_bound` are the standard outlier fences (1.5 × IQR beyond the quartiles).
- `clip()` caps any value outside the fences at the nearest fence. This keeps all rows while reducing the effect of extreme values.
- `print(df)` displays the DataFrame after outlier capping.

### 6. Checking missing value percentage
```python
print(df.isnull().sum()/len(df)*100)
```
- Shows the percentage of missing values per column, which helps decide how to treat them.

### 7. KNN imputation of missing values
```python
imputer = KNNImputer(n_neighbors=5)
df[[col]] = imputer.fit_transform(df[[col]])
print(df.isnull().sum())
```
- Creates a KNN imputer that uses the 5 nearest neighbours.
- `fit_transform` fills the missing values and writes the result back into the DataFrame.
- The final `print` confirms the remaining missing values.

### 8. Handling missing coupon codes
```python
df['CouponCode'] = df['CouponCode'].fillna('Not applicable')
```
- A missing coupon code means no coupon was used, so these blanks are filled with `'Not applicable'`.

### 9. One-hot encoding categorical columns
```python
df = pd.get_dummies(df, columns=['PaymentMethod', 'OrderStatus', 'ShippingAddress', 'ReferralSource'])
```
- Converts each categorical column into separate True/False columns, one per category (e.g., `PaymentMethod_Cash`, `OrderStatus_Delivered`).
- Machine learning models need numeric input, so this makes the categories usable.
- Because `ShippingAddress` has hundreds of unique values, this step creates most of the 680 columns in the output.

### 10. Removing highly correlated features
```python
corr_matrix = df.corr(numeric_only=True)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
df = df.drop(columns=to_drop)
```
- `df.corr(numeric_only=True)` builds the correlation matrix.
- `np.triu(..., k=1)` keeps only the upper triangle, so each pair of columns is checked once.
- `to_drop` lists every column with a correlation above 0.8 with another column.
- `df.drop(columns=to_drop)` removes those columns to reduce redundancy and multicollinearity.

### 11. Schema validation
```python
schema = DataFrameSchema({'ItemsInCart': Column(pa.Int), 'TotalPrice': Column(pa.Float)})
schema.validate(df)
```
- Defines the expected data types: `ItemsInCart` must be an integer and `TotalPrice` must be a float.
- `schema.validate(df)` raises an error if the data does not match, which acts as a final quality check.

### 12. Saving the cleaned data
```python
df.to_csv('/Users/apurvgaurav/Downloads/week 1_cleaned.csv', index=False)
```
- Writes the cleaned DataFrame to a new CSV file without the index column.

## Results

| | Raw Data | Cleaned Data |
|---|---|---|
| Rows | 1,200 | 1,200 |
| Columns | 14 | 680 |
| Outliers | Present | Capped using IQR |
| Missing `CouponCode` | Blank | `Not applicable` |
| Categorical columns | Text | One-hot encoded (Boolean) |

## How to Run

1. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn pandera
   ```
2. Update the file paths in `data cleaning.py` to point to your local `week 1.csv`.
3. Run the script:
   ```bash
   python "data cleaning.py"
   ```
4. The cleaned file will be saved as `week 1_cleaned.csv`.

## Possible Improvements

- Remove duplicate rows with `df.drop_duplicates()` after detecting them.
- Run the KNN imputer on all numeric columns together, inside or after the loop, instead of on a single column.
- Group rare `ShippingAddress` values, or drop the column, instead of one-hot encoding it, to avoid hundreds of sparse columns.
- Convert `Date` to a proper datetime type.
- Use `pa.Int64` in the schema, since `pa.Int` is deprecated in newer Pandera versions.

## Author

**Apurv Gaurav**

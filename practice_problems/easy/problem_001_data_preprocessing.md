# Problem 001: Data Preprocessing Basics

## Problem Statement

You are given a dataset with the following issues:
1. Missing values in multiple columns
2. Different scales for numerical features
3. Categorical variables that need encoding
4. Duplicate rows

Write a Python function to preprocess this data for machine learning.

## Input Data Example
```python
import pandas as pd
import numpy as np

# Sample dataset with issues
data = {
    'age': [25, 30, np.nan, 45, 25, 35, np.nan, 40],
    'income': [50000, 60000, 45000, np.nan, 50000, 80000, 75000, 90000],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Bachelor', 'Master', 'PhD', np.nan],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'NYC', 'Boston', 'Seattle', 'LA'],
    'score': [0.8, 0.9, 0.7, 0.85, 0.8, 0.95, 0.88, 0.92]
}

df = pd.DataFrame(data)
```

## Requirements

Implement a function `preprocess_data(df)` that:
1. Handles missing values appropriately
2. Scales numerical features to [0, 1] range
3. Encodes categorical variables
4. Removes duplicate rows
5. Returns the preprocessed dataset

## Expected Function Signature
```python
def preprocess_data(df):
    """
    Preprocess the input dataframe for machine learning
    
    Args:
        df (pd.DataFrame): Input dataframe with potential data quality issues
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for ML
    """
    # Your implementation here
    pass
```

## Constraints
- Handle missing values using appropriate strategies
- Use MinMaxScaler for numerical features
- Use one-hot encoding for categorical features
- Maintain the original number of features semantically

## Example Output
The function should return a clean dataframe with:
- No missing values
- All numerical columns scaled between 0 and 1
- Categorical columns properly encoded
- No duplicate rows

---

# Solution

<details>
<summary>Click to see solution</summary>

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Preprocess the input dataframe for machine learning
    
    Args:
        df (pd.DataFrame): Input dataframe with potential data quality issues
        
    Returns:
        pd.DataFrame: Preprocessed dataframe ready for ML
    """
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Step 1: Remove duplicate rows
    df_processed = df_processed.drop_duplicates()
    
    # Step 2: Separate numerical and categorical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Step 3: Handle missing values
    # For numerical columns: use median imputation
    if numerical_cols:
        num_imputer = SimpleImputer(strategy='median')
        df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
    
    # For categorical columns: use most frequent value imputation
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
    
    # Step 4: Scale numerical features to [0, 1]
    if numerical_cols:
        scaler = MinMaxScaler()
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    
    # Step 5: One-hot encode categorical variables
    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df_processed

# Test the solution
def test_solution():
    # Create test data
    data = {
        'age': [25, 30, np.nan, 45, 25, 35, np.nan, 40],
        'income': [50000, 60000, 45000, np.nan, 50000, 80000, 75000, 90000],
        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Bachelor', 'Master', 'PhD', np.nan],
        'city': ['NYC', 'LA', 'Chicago', 'NYC', 'NYC', 'Boston', 'Seattle', 'LA'],
        'score': [0.8, 0.9, 0.7, 0.85, 0.8, 0.95, 0.88, 0.92]
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nOriginal shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    print("\nProcessed DataFrame:")
    print(df_processed)
    print(f"\nProcessed shape: {df_processed.shape}")
    print(f"Missing values after processing:\n{df_processed.isnull().sum()}")
    
    # Check if numerical columns are scaled between 0 and 1
    numerical_cols = ['age', 'income', 'score']
    for col in numerical_cols:
        if col in df_processed.columns:
            print(f"\n{col} range: [{df_processed[col].min():.3f}, {df_processed[col].max():.3f}]")

# Run the test
if __name__ == "__main__":
    test_solution()
```

</details>

## Alternative Solutions

<details>
<summary>Advanced Solution with Pipeline</summary>

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_preprocessing_pipeline(df):
    """
    Create a sklearn pipeline for preprocessing
    """
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Create preprocessing pipelines for each data type
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor

def preprocess_data_pipeline(df):
    """
    Preprocess data using sklearn pipeline
    """
    # Remove duplicates first
    df_no_duplicates = df.drop_duplicates()
    
    # Create and fit preprocessor
    preprocessor = create_preprocessing_pipeline(df_no_duplicates)
    processed_data = preprocessor.fit_transform(df_no_duplicates)
    
    # Get feature names after preprocessing
    numerical_cols = df_no_duplicates.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_no_duplicates.select_dtypes(include=['object']).columns.tolist()
    
    # Get encoded categorical column names
    cat_encoder = preprocessor.named_transformers_['cat']['encoder']
    encoded_cat_names = []
    for i, col in enumerate(categorical_cols):
        encoded_names = [f"{col}_{cat}" for cat in cat_encoder.categories_[i][1:]]  # Skip first due to drop='first'
        encoded_cat_names.extend(encoded_names)
    
    feature_names = numerical_cols + encoded_cat_names
    
    # Convert back to DataFrame
    processed_df = pd.DataFrame(processed_data, columns=feature_names, index=df_no_duplicates.index)
    
    return processed_df
```

</details>

## Learning Objectives

After solving this problem, you should understand:
1. How to handle missing values with different strategies
2. Feature scaling techniques and when to apply them
3. Categorical variable encoding methods
4. Data cleaning best practices
5. Building preprocessing pipelines

## Common Interview Follow-up Questions

1. **Q: Why did you choose median imputation over mean imputation?**
   A: Median is more robust to outliers and skewed distributions.

2. **Q: When would you use StandardScaler vs MinMaxScaler?**
   A: StandardScaler for normally distributed data, MinMaxScaler when you need bounded values.

3. **Q: What are the disadvantages of one-hot encoding?**
   A: Creates many features with high cardinality, can lead to curse of dimensionality.

4. **Q: How would you handle this in a production pipeline?**
   A: Use sklearn pipelines, save fitted transformers, implement proper validation.

## Difficulty: Easy
**Time to solve:** 15-30 minutes  
**Key concepts:** Data preprocessing, missing values, feature scaling, encoding

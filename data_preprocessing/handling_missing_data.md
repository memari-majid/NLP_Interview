# Handling Missing Data

## Overview
Missing data is a common challenge in real-world datasets. Proper handling of missing values is crucial for building robust machine learning models.

## Types of Missing Data

### 1. Missing Completely at Random (MCAR)
- Missing values are independent of both observed and unobserved data
- The probability of missing is the same for all observations
- **Example**: Survey responses lost due to technical issues

### 2. Missing at Random (MAR)
- Missing values depend on observed data but not on the missing values themselves
- Can be predicted from other variables
- **Example**: Income data missing for certain age groups

### 3. Missing Not at Random (MNAR)
- Missing values depend on the unobserved values themselves
- Systematic missingness based on the value that would have been observed
- **Example**: High earners refusing to disclose salary information

## Detection Strategies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_data(df):
    """Comprehensive missing data analysis"""
    
    # Calculate missing counts and percentages
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentages.values
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    print("Missing Data Summary:")
    print(missing_df.to_string(index=False))
    
    # Visualize missing data patterns
    plt.figure(figsize=(12, 8))
    
    # Missing data heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Data Heatmap')
    
    # Missing data bar plot
    plt.subplot(2, 2, 2)
    missing_df.plot(x='Column', y='Missing_Percentage', kind='bar', ax=plt.gca())
    plt.title('Missing Data Percentage by Column')
    plt.xticks(rotation=45)
    
    # Missing data correlation
    plt.subplot(2, 2, 3)
    missing_corr = df.isnull().corr()
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Missing Data Correlation')
    
    plt.tight_layout()
    plt.show()
    
    return missing_df

# Example usage
# missing_analysis = analyze_missing_data(df)
```

## Handling Techniques

### 1. Deletion Methods

#### Complete Case Analysis (Listwise Deletion)
```python
def complete_case_deletion(df):
    """Remove rows with any missing values"""
    return df.dropna()

def pairwise_deletion(df, columns):
    """Remove rows with missing values in specific columns"""
    return df.dropna(subset=columns)
```

**Pros:** Simple, unbiased if MCAR
**Cons:** Reduces sample size, may introduce bias if not MCAR

#### Threshold-Based Deletion
```python
def threshold_deletion(df, row_threshold=0.5, col_threshold=0.5):
    """
    Remove rows/columns based on missing data threshold
    """
    # Remove rows with more than row_threshold missing
    row_mask = df.isnull().mean(axis=1) <= row_threshold
    df_cleaned = df[row_mask]
    
    # Remove columns with more than col_threshold missing
    col_mask = df_cleaned.isnull().mean(axis=0) <= col_threshold
    df_cleaned = df_cleaned.loc[:, col_mask]
    
    return df_cleaned
```

### 2. Imputation Methods

#### Statistical Imputation
```python
from sklearn.impute import SimpleImputer

class StatisticalImputer:
    def __init__(self, strategy='mean'):
        """
        strategy: 'mean', 'median', 'most_frequent', 'constant'
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.fill_values = {}
    
    def fit_transform(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df_imputed = df.copy()
        
        # Handle numeric columns
        if len(numeric_cols) > 0:
            if self.strategy == 'mean':
                self.fill_values.update({col: df[col].mean() for col in numeric_cols})
            elif self.strategy == 'median':
                self.fill_values.update({col: df[col].median() for col in numeric_cols})
        
        # Handle categorical columns
        if len(categorical_cols) > 0:
            self.fill_values.update({col: df[col].mode()[0] for col in categorical_cols})
        
        # Fill missing values
        df_imputed = df_imputed.fillna(self.fill_values)
        
        return df_imputed

# Example usage
imputer = StatisticalImputer(strategy='median')
df_imputed = imputer.fit_transform(df)
```

#### Forward Fill and Backward Fill (Time Series)
```python
def time_series_imputation(df, method='ffill'):
    """
    Forward fill or backward fill for time series data
    """
    if method == 'ffill':
        return df.fillna(method='ffill')
    elif method == 'bfill':
        return df.fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')

# Combine methods
def combined_fill(df):
    return df.fillna(method='ffill').fillna(method='bfill')
```

#### Advanced Imputation Techniques

##### K-Nearest Neighbors (KNN) Imputation
```python
from sklearn.impute import KNNImputer

def knn_imputation(df, n_neighbors=5):
    """KNN-based imputation"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df_imputed = df.copy()
    
    # Apply KNN imputation to numeric columns
    if len(numeric_cols) > 0:
        df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Handle categorical columns separately
    for col in categorical_cols:
        df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
    
    return df_imputed
```

##### Iterative Imputation (MICE)
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def iterative_imputation(df, max_iter=10, random_state=42):
    """Multiple Imputation by Chained Equations (MICE)"""
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = df.copy()
    
    for col in categorical_cols:
        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        df_encoded[col] = df_encoded[col].replace(-1, np.nan)
    
    # Apply iterative imputation
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_encoded),
        columns=df_encoded.columns,
        index=df_encoded.index
    )
    
    # Decode categorical variables
    for col in categorical_cols:
        unique_values = df[col].dropna().unique()
        df_imputed[col] = df_imputed[col].round().astype(int)
        df_imputed[col] = df_imputed[col].map(dict(enumerate(unique_values)))
    
    return df_imputed
```

### 3. Model-Based Imputation

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def model_based_imputation(df, target_col, model_type='regression'):
    """Use ML models to predict missing values"""
    
    df_imputed = df.copy()
    
    # Separate complete and incomplete cases
    complete_mask = ~df[target_col].isnull()
    incomplete_mask = df[target_col].isnull()
    
    if complete_mask.sum() == 0:
        return df_imputed  # No complete cases to train on
    
    # Prepare features (exclude target column)
    feature_cols = [col for col in df.columns if col != target_col]
    X_complete = df.loc[complete_mask, feature_cols]
    y_complete = df.loc[complete_mask, target_col]
    X_incomplete = df.loc[incomplete_mask, feature_cols]
    
    # Handle missing values in features
    feature_imputer = SimpleImputer(strategy='median')
    X_complete_imputed = feature_imputer.fit_transform(X_complete)
    X_incomplete_imputed = feature_imputer.transform(X_incomplete)
    
    # Train model
    if model_type == 'regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_complete_imputed, y_complete)
    
    # Predict missing values
    predictions = model.predict(X_incomplete_imputed)
    df_imputed.loc[incomplete_mask, target_col] = predictions
    
    return df_imputed
```

## Evaluation of Imputation Quality

```python
def evaluate_imputation(original_df, imputed_df, missing_mask):
    """Evaluate imputation quality using various metrics"""
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    for col in original_df.columns:
        if missing_mask[col].sum() > 0:  # If column had missing values
            
            # Get actual values (for evaluation, assume we artificially introduced missingness)
            actual = original_df.loc[missing_mask[col], col]
            imputed = imputed_df.loc[missing_mask[col], col]
            
            if original_df[col].dtype in ['float64', 'int64']:
                # Numeric columns
                mse = mean_squared_error(actual, imputed)
                mae = mean_absolute_error(actual, imputed)
                r2 = r2_score(actual, imputed)
                
                results[col] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R²': r2
                }
            else:
                # Categorical columns
                accuracy = (actual == imputed).mean()
                results[col] = {
                    'Accuracy': accuracy
                }
    
    return results
```

## Best Practices and Guidelines

### Decision Framework
```python
def recommend_imputation_strategy(df, column):
    """Recommend imputation strategy based on data characteristics"""
    
    missing_pct = df[column].isnull().mean() * 100
    col_type = df[column].dtype
    unique_values = df[column].nunique()
    
    recommendations = []
    
    # Based on missing percentage
    if missing_pct > 70:
        recommendations.append("Consider dropping the column")
    elif missing_pct > 40:
        recommendations.append("Use advanced imputation (MICE, model-based)")
    elif missing_pct > 10:
        recommendations.append("Use KNN or statistical imputation")
    else:
        recommendations.append("Simple imputation methods are sufficient")
    
    # Based on data type
    if col_type in ['float64', 'int64']:
        if unique_values > 10:
            recommendations.append("Use mean/median imputation or KNN")
        else:
            recommendations.append("Use mode imputation")
    else:
        recommendations.append("Use mode imputation or create 'missing' category")
    
    return recommendations
```

## Common Interview Questions

### 1. How do you handle missing data?
- Understand the type of missingness (MCAR, MAR, MNAR)
- Choose appropriate strategy based on data characteristics
- Consider the impact on model performance
- Always validate imputation quality

### 2. When would you delete vs. impute missing data?
- **Delete when:**
  - Very few missing values (< 5%)
  - Missing completely at random
  - Large dataset size
- **Impute when:**
  - Significant amount of missing data
  - Small dataset
  - Missing data has patterns

### 3. What are the pros and cons of different imputation methods?
- **Mean/Median:** Simple but reduces variance
- **KNN:** Preserves relationships but computationally expensive
- **MICE:** Handles complex patterns but time-consuming
- **Model-based:** Most accurate but may overfit

## Code Templates for Practice

```python
# Template for comprehensive missing data handling pipeline
class MissingDataHandler:
    def __init__(self, strategy='auto'):
        self.strategy = strategy
        self.imputers = {}
    
    def fit(self, X):
        # Analyze missing patterns
        # Choose optimal strategy for each column
        # Fit imputation models
        pass
    
    def transform(self, X):
        # Apply fitted imputation strategies
        pass
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

This comprehensive approach to handling missing data will help you tackle one of the most common challenges in real-world ML projects!

# Linear Regression

## Overview
Linear regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable and independent variables using a linear equation.

## Mathematical Foundation

### Simple Linear Regression
For one feature: `y = β₀ + β₁x + ε`

### Multiple Linear Regression
For multiple features: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε`

Where:
- `y`: dependent variable (target)
- `x`: independent variables (features)
- `β`: coefficients (parameters)
- `ε`: error term

## Cost Function
Mean Squared Error (MSE): `J(β) = (1/2m) Σ(hβ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²`

## Implementation from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate cost
            cost = self._calculate_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def _calculate_cost(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)
    
    def plot_cost_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = 2 * X.flatten() + 1 + np.random.randn(100) * 0.1
    
    # Train model
    model = LinearRegression(learning_rate=0.01, max_iterations=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, label='Actual')
    plt.plot(X, predictions, color='red', label='Predicted')
    plt.legend()
    plt.title('Linear Regression Results')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()
```

## Normal Equation (Analytical Solution)

For small datasets, you can solve directly:
`β = (X^T X)^(-1) X^T y`

```python
def normal_equation(X, y):
    # Add bias term
    X_with_bias = np.c_[np.ones(X.shape[0]), X]
    
    # Calculate parameters
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return theta
```

## Common Interview Questions

### 1. Assumptions of Linear Regression
- **Linearity**: Relationship between features and target is linear
- **Independence**: Observations are independent
- **Homoscedasticity**: Constant variance of residuals
- **Normality**: Residuals are normally distributed
- **No multicollinearity**: Features are not highly correlated

### 2. When to Use Linear Regression?
- Continuous target variable
- Linear relationship exists
- Need interpretable model
- Small to medium datasets
- Baseline model for comparison

### 3. Advantages and Disadvantages

**Advantages:**
- Simple and fast
- Interpretable coefficients
- No hyperparameter tuning
- Good baseline model
- Works well with linear relationships

**Disadvantages:**
- Assumes linear relationship
- Sensitive to outliers
- Requires feature scaling for gradient descent
- Poor performance with non-linear data

## Regularization Variants

### Ridge Regression (L2)
Cost function: `J(β) = MSE + α Σβᵢ²`

### Lasso Regression (L1)
Cost function: `J(β) = MSE + α Σ|βᵢ|`

### Elastic Net
Cost function: `J(β) = MSE + α₁ Σ|βᵢ| + α₂ Σβᵢ²`

## Performance Metrics

```python
def calculate_metrics(y_true, y_pred):
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
```

## Practice Problems

1. **Implement Ridge Regression from scratch**
2. **Handle multicollinearity in linear regression**
3. **Implement polynomial regression using linear regression**
4. **Add regularization to prevent overfitting**
5. **Implement cross-validation for model selection**

## Key Interview Tips

- Always explain the mathematical foundation
- Discuss assumptions and when they might be violated
- Compare with other regression techniques
- Explain regularization and its importance
- Demonstrate with code implementation

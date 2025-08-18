# Problem 001: Implement K-Nearest Neighbors from Scratch

## Problem Statement

Implement the K-Nearest Neighbors (KNN) algorithm from scratch for both classification and regression tasks. Your implementation should support different distance metrics and handle edge cases properly.

## Requirements

Implement a `KNNClassifier` and `KNNRegressor` class with the following specifications:

### KNNClassifier
```python
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN Classifier
        
        Args:
            k (int): Number of neighbors to consider
            distance_metric (str): 'euclidean', 'manhattan', or 'cosine'
        """
        pass
    
    def fit(self, X, y):
        """Store training data"""
        pass
    
    def predict(self, X):
        """Predict class labels for test data"""
        pass
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        pass
```

### KNNRegressor
```python
class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        Initialize KNN Regressor
        
        Args:
            k (int): Number of neighbors to consider
            distance_metric (str): Distance metric to use
            weights (str): 'uniform' or 'distance' weighted
        """
        pass
    
    def fit(self, X, y):
        """Store training data"""
        pass
    
    def predict(self, X):
        """Predict target values for test data"""
        pass
```

## Input/Output Examples

### Classification Example
```python
# Training data
X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Test data
X_test = np.array([[2, 2], [7, 6]])

# Expected behavior
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)  # Should return [0, 1]
probabilities = knn.predict_proba(X_test)  # Should return probabilities
```

### Regression Example
```python
# Training data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([2, 4, 6, 8, 10])

# Test data
X_test = np.array([[2.5], [4.5]])

# Expected behavior
knn = KNNRegressor(k=3, weights='distance')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)  # Should return [5.0, 9.0] approximately
```

## Constraints and Edge Cases

1. Handle cases where k > number of training samples
2. Support different distance metrics efficiently
3. Handle tie-breaking in classification
4. Implement both uniform and distance-weighted predictions for regression
5. Handle empty neighbor sets gracefully
6. Support multi-dimensional feature spaces

## Distance Metrics to Implement

1. **Euclidean**: √(Σ(xi - yi)²)
2. **Manhattan**: Σ|xi - yi|
3. **Cosine**: 1 - (x·y)/(||x||||y||)

---

# Solution

<details>
<summary>Click to see solution</summary>

```python
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist

class KNNBase:
    """Base class for KNN implementations"""
    
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def _calculate_distances(self, X_test):
        """Calculate distances between test and training points"""
        if self.distance_metric == 'euclidean':
            return cdist(X_test, self.X_train, metric='euclidean')
        elif self.distance_metric == 'manhattan':
            return cdist(X_test, self.X_train, metric='manhattan')
        elif self.distance_metric == 'cosine':
            return cdist(X_test, self.X_train, metric='cosine')
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _get_neighbors(self, distances, k):
        """Get k nearest neighbors for each test point"""
        # Handle case where k > number of training samples
        effective_k = min(k, len(self.X_train))
        
        # Get indices of k nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, :effective_k]
        
        # Get corresponding distances
        neighbor_distances = np.array([
            distances[i, neighbor_indices[i]] 
            for i in range(len(distances))
        ])
        
        return neighbor_indices, neighbor_distances
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

class KNNClassifier(KNNBase):
    """K-Nearest Neighbors Classifier"""
    
    def __init__(self, k=3, distance_metric='euclidean'):
        super().__init__(k, distance_metric)
        self.classes_ = None
    
    def fit(self, X, y):
        """Store training data and extract unique classes"""
        super().fit(X, y)
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        """Predict class labels for test data"""
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_test = np.array(X)
        
        # Calculate distances
        distances = self._calculate_distances(X_test)
        
        # Get neighbors
        neighbor_indices, _ = self._get_neighbors(distances, self.k)
        
        predictions = []
        
        for neighbors in neighbor_indices:
            # Get neighbor labels
            neighbor_labels = self.y_train[neighbors]
            
            # Vote for most common class (handle ties by taking first)
            vote_counts = Counter(neighbor_labels)
            predicted_class = vote_counts.most_common(1)[0][0]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_test = np.array(X)
        
        # Calculate distances
        distances = self._calculate_distances(X_test)
        
        # Get neighbors
        neighbor_indices, _ = self._get_neighbors(distances, self.k)
        
        probabilities = []
        
        for neighbors in neighbor_indices:
            # Get neighbor labels
            neighbor_labels = self.y_train[neighbors]
            
            # Calculate probabilities for each class
            class_probs = {}
            for class_label in self.classes_:
                class_probs[class_label] = np.sum(neighbor_labels == class_label) / len(neighbors)
            
            # Convert to ordered probability array
            prob_array = [class_probs[class_label] for class_label in self.classes_]
            probabilities.append(prob_array)
        
        return np.array(probabilities)

class KNNRegressor(KNNBase):
    """K-Nearest Neighbors Regressor"""
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        super().__init__(k, distance_metric)
        self.weights = weights
    
    def predict(self, X):
        """Predict target values for test data"""
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_test = np.array(X)
        
        # Calculate distances
        distances = self._calculate_distances(X_test)
        
        # Get neighbors
        neighbor_indices, neighbor_distances = self._get_neighbors(distances, self.k)
        
        predictions = []
        
        for i, neighbors in enumerate(neighbor_indices):
            # Get neighbor target values
            neighbor_values = self.y_train[neighbors]
            
            if self.weights == 'uniform':
                # Simple average
                prediction = np.mean(neighbor_values)
            elif self.weights == 'distance':
                # Distance-weighted average
                distances_to_neighbors = neighbor_distances[i]
                
                # Handle case where distance is 0 (exact match)
                if np.any(distances_to_neighbors == 0):
                    # If any neighbor has distance 0, use only those neighbors
                    zero_distance_mask = distances_to_neighbors == 0
                    prediction = np.mean(neighbor_values[zero_distance_mask])
                else:
                    # Weight by inverse distance
                    weights = 1.0 / distances_to_neighbors
                    weights /= np.sum(weights)  # Normalize weights
                    prediction = np.sum(weights * neighbor_values)
            else:
                raise ValueError(f"Unsupported weight type: {self.weights}")
            
            predictions.append(prediction)
        
        return np.array(predictions)

# Alternative implementation without scipy (from scratch)
class KNNFromScratch:
    """Complete KNN implementation without external distance libraries"""
    
    @staticmethod
    def euclidean_distance(x1, x2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def manhattan_distance(x1, x2):
        """Calculate Manhattan distance between two points"""
        return np.sum(np.abs(x1 - x2))
    
    @staticmethod
    def cosine_distance(x1, x2):
        """Calculate Cosine distance between two points"""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        return 1 - cosine_similarity
    
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.distance_functions = {
            'euclidean': self.euclidean_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_distance
        }
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _calculate_distances(self, x_test):
        """Calculate distances from a test point to all training points"""
        distances = []
        distance_func = self.distance_functions[self.distance_metric]
        
        for x_train in self.X_train:
            distance = distance_func(x_test, x_train)
            distances.append(distance)
        
        return np.array(distances)
    
    def predict_single(self, x_test):
        """Predict for a single test point"""
        distances = self._calculate_distances(x_test)
        
        # Get k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # For classification: return most common label
        # For regression: return mean (implement as needed)
        if len(np.unique(self.y_train)) < len(self.y_train) // 2:  # Likely classification
            return Counter(k_nearest_labels).most_common(1)[0][0]
        else:  # Likely regression
            return np.mean(k_nearest_labels)
    
    def predict(self, X):
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)

# Test functions
def test_knn_classifier():
    """Test KNN Classifier"""
    print("Testing KNN Classifier...")
    
    # Create sample data
    X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 2], [7, 6]])
    
    # Test classifier
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test)
    probabilities = knn.predict_proba(X_test)
    
    print(f"Predictions: {predictions}")
    print(f"Probabilities:\n{probabilities}")
    
    return predictions, probabilities

def test_knn_regressor():
    """Test KNN Regressor"""
    print("\nTesting KNN Regressor...")
    
    # Create sample data
    X_train = np.array([[1], [2], [3], [4], [5]])
    y_train = np.array([2, 4, 6, 8, 10])
    X_test = np.array([[2.5], [4.5]])
    
    # Test uniform weights
    knn_uniform = KNNRegressor(k=3, weights='uniform')
    knn_uniform.fit(X_train, y_train)
    predictions_uniform = knn_uniform.predict(X_test)
    
    # Test distance weights
    knn_distance = KNNRegressor(k=3, weights='distance')
    knn_distance.fit(X_train, y_train)
    predictions_distance = knn_distance.predict(X_test)
    
    print(f"Uniform weights predictions: {predictions_uniform}")
    print(f"Distance weights predictions: {predictions_distance}")
    
    return predictions_uniform, predictions_distance

if __name__ == "__main__":
    test_knn_classifier()
    test_knn_regressor()
```

</details>

## Performance Analysis

<details>
<summary>Time and Space Complexity Analysis</summary>

### Time Complexity:
- **Training (fit)**: O(1) - just storing the data
- **Prediction**: O(n * m * d) where:
  - n = number of test samples
  - m = number of training samples  
  - d = number of features (for distance calculation)
- **Overall**: O(n * m * d + n * m * log(m)) including sorting for k-nearest

### Space Complexity:
- **Training data storage**: O(m * d)
- **Distance matrix**: O(n * m)
- **Overall**: O(m * d + n * m)

### Optimization Techniques:
1. **KD-Trees** for faster nearest neighbor search: O(log m) per query
2. **Ball Trees** for high-dimensional spaces
3. **LSH (Locality Sensitive Hashing)** for approximate neighbors
4. **Approximate methods** for very large datasets

</details>

## Extensions and Variations

<details>
<summary>Advanced KNN Features</summary>

```python
class AdvancedKNN:
    """KNN with additional features"""
    
    def __init__(self, k=3, distance_metric='euclidean', 
                 algorithm='brute', leaf_size=30):
        self.k = k
        self.distance_metric = distance_metric
        self.algorithm = algorithm  # 'brute', 'kd_tree', 'ball_tree'
        self.leaf_size = leaf_size
    
    def fit_with_validation(self, X, y, X_val, y_val):
        """Fit with automatic k selection using validation set"""
        best_k = 1
        best_score = 0
        
        for k in range(1, min(21, len(X))):
            temp_knn = KNNClassifier(k=k, distance_metric=self.distance_metric)
            temp_knn.fit(X, y)
            
            val_predictions = temp_knn.predict(X_val)
            accuracy = np.mean(val_predictions == y_val)
            
            if accuracy > best_score:
                best_score = accuracy
                best_k = k
        
        self.k = best_k
        return self.fit(X, y)
    
    def feature_importance(self, X, y, n_iterations=10):
        """Calculate feature importance using permutation"""
        baseline_score = self._cross_validate_score(X, y)
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            scores = []
            
            for _ in range(n_iterations):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, feature_idx])
                
                permuted_score = self._cross_validate_score(X_permuted, y)
                importance = baseline_score - permuted_score
                scores.append(importance)
            
            importance_scores.append(np.mean(scores))
        
        return np.array(importance_scores)
```

</details>

## Common Interview Follow-up Questions

1. **Q: What are the advantages and disadvantages of KNN?**
   
   **Advantages:**
   - Simple to understand and implement
   - No assumptions about data distribution
   - Works well with small datasets
   - Can be used for both classification and regression
   
   **Disadvantages:**
   - Computationally expensive for large datasets
   - Sensitive to irrelevant features
   - Requires feature scaling
   - Sensitive to local structure of data

2. **Q: How do you choose the optimal value of k?**
   - Use cross-validation
   - Odd k for binary classification (avoids ties)
   - Start with k = √n as rule of thumb
   - Plot validation accuracy vs k

3. **Q: When would you use KNN over other algorithms?**
   - Small to medium datasets
   - Non-linear decision boundaries
   - Need local patterns
   - Baseline model for comparison

4. **Q: How would you optimize KNN for large datasets?**
   - Use approximate nearest neighbor algorithms
   - Implement KD-trees or Ball trees
   - Use dimensionality reduction
   - Sample the training data

## Learning Objectives

After solving this problem, you should understand:
- KNN algorithm implementation details
- Different distance metrics and their properties
- Handling edge cases in ML algorithms
- Time and space complexity trade-offs
- Weighted vs uniform voting strategies

## Difficulty: Medium
**Time to solve:** 45-90 minutes  
**Key concepts:** Distance metrics, nearest neighbors, classification vs regression, algorithm optimization

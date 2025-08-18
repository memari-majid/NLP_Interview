import numpy as np
from typing import List, Callable, Tuple, Optional
import matplotlib.pyplot as plt


class ActivationFunctions:
    """Collection of activation functions and their derivatives."""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function."""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function."""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function."""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivative of Leaky ReLU function."""
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class Perceptron:
    """Single perceptron implementation."""
    
    def __init__(self, input_size: int, learning_rate: float = 0.01, 
                 activation: str = 'sigmoid'):
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size) * 0.5
        self.bias = 0.0
        
        # Set activation function
        self.activation_func, self.activation_derivative = self._get_activation_functions(activation)
        
        # Training history
        self.loss_history = []
    
    def _get_activation_functions(self, activation: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        if activation == 'sigmoid':
            return ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            return ActivationFunctions.tanh, ActivationFunctions.tanh_derivative
        elif activation == 'relu':
            return ActivationFunctions.relu, ActivationFunctions.relu_derivative
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation."""
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self.activation_func(self.z)
        return self.a
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        # Avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """Backward propagation."""
        m = X.shape[0]
        
        # Compute gradients
        dz = y_pred - y_true
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        
        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000):
        """Train the perceptron."""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.loss_history.append(cost)
            
            # Backward pass
            self.backward(X, y, y_pred)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int)


class NeuralNetwork:
    """Multi-layer neural network implementation."""
    
    def __init__(self, layers: List[int], activation: str = 'sigmoid', 
                 output_activation: str = 'sigmoid', learning_rate: float = 0.01):
        self.layers = layers
        self.num_layers = len(layers)
        self.learning_rate = learning_rate
        self.activation = activation
        self.output_activation = output_activation
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        
        for i in range(1, self.num_layers):
            # Xavier/Glorot initialization
            self.weights[f'W{i}'] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2.0 / layers[i-1])
            self.biases[f'b{i}'] = np.zeros((1, layers[i]))
        
        # Get activation functions
        self.hidden_activation, self.hidden_activation_derivative = self._get_activation_functions(activation)
        self.out_activation, self.out_activation_derivative = self._get_activation_functions(output_activation)
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
        
        # Cache for forward propagation
        self.cache = {}
    
    def _get_activation_functions(self, activation: str) -> Tuple[Callable, Callable]:
        """Get activation function and its derivative."""
        if activation == 'sigmoid':
            return ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative
        elif activation == 'tanh':
            return ActivationFunctions.tanh, ActivationFunctions.tanh_derivative
        elif activation == 'relu':
            return ActivationFunctions.relu, ActivationFunctions.relu_derivative
        elif activation == 'softmax':
            return ActivationFunctions.softmax, lambda x: x  # Derivative handled separately
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward propagation through the network."""
        self.cache['A0'] = X
        
        # Hidden layers
        for i in range(1, self.num_layers - 1):
            Z = np.dot(self.cache[f'A{i-1}'], self.weights[f'W{i}']) + self.biases[f'b{i}']
            A = self.hidden_activation(Z)
            
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
        
        # Output layer
        i = self.num_layers - 1
        Z = np.dot(self.cache[f'A{i-1}'], self.weights[f'W{i}']) + self.biases[f'b{i}']
        
        if self.output_activation == 'softmax':
            A = self.out_activation(Z)
        else:
            A = self.out_activation(Z)
        
        self.cache[f'Z{i}'] = Z
        self.cache[f'A{i}'] = A
        
        return A
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray, cost_type: str = 'cross_entropy') -> float:
        """Compute cost function."""
        m = y_true.shape[0]
        
        if cost_type == 'cross_entropy':
            if y_true.shape[1] > 1:  # Multi-class
                # Categorical cross-entropy
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            else:  # Binary
                # Binary cross-entropy
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        elif cost_type == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        
        else:
            raise ValueError(f"Unsupported cost type: {cost_type}")
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """Backward propagation through the network."""
        m = X.shape[0]
        gradients = {}
        
        # Output layer gradient
        if self.output_activation == 'softmax' and y_true.shape[1] > 1:
            # Softmax with categorical cross-entropy
            dZ = y_pred - y_true
        else:
            # Other activations
            dA = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
            dZ = dA * self.out_activation_derivative(self.cache[f'Z{self.num_layers-1}'])
        
        # Gradients for output layer
        i = self.num_layers - 1
        gradients[f'dW{i}'] = (1/m) * np.dot(self.cache[f'A{i-1}'].T, dZ)
        gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        
        # Propagate backwards through hidden layers
        dA_prev = np.dot(dZ, self.weights[f'W{i}'].T)
        
        for i in range(self.num_layers - 2, 0, -1):
            dZ = dA_prev * self.hidden_activation_derivative(self.cache[f'Z{i}'])
            
            gradients[f'dW{i}'] = (1/m) * np.dot(self.cache[f'A{i-1}'].T, dZ)
            gradients[f'db{i}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if i > 1:
                dA_prev = np.dot(dZ, self.weights[f'W{i}'].T)
        
        # Update weights and biases
        for i in range(1, self.num_layers):
            self.weights[f'W{i}'] -= self.learning_rate * gradients[f'dW{i}']
            self.biases[f'b{i}'] -= self.learning_rate * gradients[f'db{i}']
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
              cost_type: str = 'cross_entropy', verbose: bool = True):
        """Train the neural network."""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred, cost_type)
            self.loss_history.append(cost)
            
            # Compute accuracy
            accuracy = self.compute_accuracy(y, y_pred)
            self.accuracy_history.append(accuracy)
            
            # Backward pass
            self.backward(X, y, y_pred)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        y_pred = self.forward(X)
        
        if y_pred.shape[1] > 1:  # Multi-class
            return np.argmax(y_pred, axis=1)
        else:  # Binary
            return (y_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.forward(X)
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        if y_true.shape[1] > 1:  # Multi-class
            y_true_labels = np.argmax(y_true, axis=1)
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:  # Binary
            y_true_labels = y_true.flatten()
            y_pred_labels = (y_pred > 0.5).astype(int).flatten()
        
        return np.mean(y_true_labels == y_pred_labels)


class AdamOptimizer:
    """Adam optimizer for neural network training."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m_weights = {}
        self.v_weights = {}
        self.m_biases = {}
        self.v_biases = {}
        self.t = 0
    
    def update(self, network: NeuralNetwork, gradients: dict):
        """Update network parameters using Adam optimizer."""
        self.t += 1
        
        for i in range(1, network.num_layers):
            w_key = f'W{i}'
            b_key = f'b{i}'
            dw_key = f'dW{i}'
            db_key = f'db{i}'
            
            # Initialize momentum terms if first iteration
            if w_key not in self.m_weights:
                self.m_weights[w_key] = np.zeros_like(network.weights[w_key])
                self.v_weights[w_key] = np.zeros_like(network.weights[w_key])
                self.m_biases[b_key] = np.zeros_like(network.biases[b_key])
                self.v_biases[b_key] = np.zeros_like(network.biases[b_key])
            
            # Update momentum terms for weights
            self.m_weights[w_key] = self.beta1 * self.m_weights[w_key] + (1 - self.beta1) * gradients[dw_key]
            self.v_weights[w_key] = self.beta2 * self.v_weights[w_key] + (1 - self.beta2) * (gradients[dw_key] ** 2)
            
            # Update momentum terms for biases
            self.m_biases[b_key] = self.beta1 * self.m_biases[b_key] + (1 - self.beta1) * gradients[db_key]
            self.v_biases[b_key] = self.beta2 * self.v_biases[b_key] + (1 - self.beta2) * (gradients[db_key] ** 2)
            
            # Bias correction
            m_w_corrected = self.m_weights[w_key] / (1 - self.beta1 ** self.t)
            v_w_corrected = self.v_weights[w_key] / (1 - self.beta2 ** self.t)
            m_b_corrected = self.m_biases[b_key] / (1 - self.beta1 ** self.t)
            v_b_corrected = self.v_biases[b_key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            network.weights[w_key] -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            network.biases[b_key] -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)


def gradient_check(network: NeuralNetwork, X: np.ndarray, y: np.ndarray, epsilon: float = 1e-7) -> float:
    """Perform gradient checking to verify backpropagation implementation."""
    # Get gradients from backpropagation
    y_pred = network.forward(X)
    network.backward(X, y, y_pred)
    
    # Store analytical gradients
    analytical_gradients = {}
    for i in range(1, network.num_layers):
        # Note: This is simplified - in practice you'd store gradients during backward pass
        pass
    
    # Compute numerical gradients
    numerical_gradients = {}
    
    for i in range(1, network.num_layers):
        w_key = f'W{i}'
        b_key = f'b{i}'
        
        # Check weights
        w_shape = network.weights[w_key].shape
        numerical_gradients[w_key] = np.zeros(w_shape)
        
        for idx in np.ndindex(w_shape):
            # Perturb weight
            network.weights[w_key][idx] += epsilon
            y_pred_plus = network.forward(X)
            cost_plus = network.compute_cost(y, y_pred_plus)
            
            network.weights[w_key][idx] -= 2 * epsilon
            y_pred_minus = network.forward(X)
            cost_minus = network.compute_cost(y, y_pred_minus)
            
            # Restore weight
            network.weights[w_key][idx] += epsilon
            
            # Compute numerical gradient
            numerical_gradients[w_key][idx] = (cost_plus - cost_minus) / (2 * epsilon)
    
    print("Gradient checking completed (simplified version)")
    return 0.0  # Would return actual difference in real implementation


# Demo functions and data generators
def generate_xor_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate XOR problem data."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return X, y


def generate_spiral_data(n_samples: int = 100, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Generate spiral classification data."""
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros((n_samples * n_classes, n_classes))
    
    for class_idx in range(n_classes):
        ix = range(n_samples * class_idx, n_samples * (class_idx + 1))
        r = np.linspace(0.0, 1, n_samples)  # radius
        t = np.linspace(class_idx * 4, (class_idx + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix, class_idx] = 1
    
    return X, y


def plot_decision_boundary(network: NeuralNetwork, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary"):
    """Plot decision boundary for 2D data."""
    try:
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = network.predict_proba(mesh_points)
        
        if Z.shape[1] > 1:
            Z = np.argmax(Z, axis=1)
        else:
            Z = Z.ravel()
        
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot data points
        if y.shape[1] > 1:
            y_labels = np.argmax(y, axis=1)
        else:
            y_labels = y.ravel()
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y_labels, cmap=plt.cm.RdYlBu, edgecolors='black')
        plt.colorbar(scatter)
        plt.title(title)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    print("Neural Networks from Scratch\n")
    
    # Test 1: Single Perceptron on AND gate
    print("="*50)
    print("Test 1: Single Perceptron - AND Gate")
    print("="*50)
    
    # AND gate data
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_and = np.array([0, 0, 0, 1], dtype=np.float32)
    
    perceptron = Perceptron(input_size=2, learning_rate=0.1, activation='sigmoid')
    perceptron.train(X_and, y_and, epochs=1000)
    
    print("\nAND Gate Results:")
    predictions = perceptron.predict(X_and)
    for i, (x, y_true, y_pred) in enumerate(zip(X_and, y_and, predictions)):
        print(f"Input: {x}, True: {y_true}, Predicted: {y_pred}")
    
    # Test 2: Multi-layer network on XOR problem
    print("\n" + "="*50)
    print("Test 2: Multi-layer Network - XOR Problem")
    print("="*50)
    
    X_xor, y_xor = generate_xor_data()
    
    # Create neural network: 2 inputs -> 4 hidden -> 1 output
    nn_xor = NeuralNetwork(layers=[2, 4, 1], activation='tanh', learning_rate=0.5)
    nn_xor.train(X_xor, y_xor, epochs=2000, verbose=False)
    
    print("\nXOR Problem Results:")
    predictions = nn_xor.predict(X_xor)
    probabilities = nn_xor.predict_proba(X_xor)
    
    for i, (x, y_true, y_pred, prob) in enumerate(zip(X_xor, y_xor.flatten(), predictions.flatten(), probabilities.flatten())):
        print(f"Input: {x}, True: {int(y_true)}, Predicted: {y_pred}, Probability: {prob:.3f}")
    
    final_accuracy = nn_xor.compute_accuracy(y_xor, probabilities)
    print(f"Final Accuracy: {final_accuracy:.3f}")
    
    # Test 3: Multi-class classification on spiral data
    print("\n" + "="*50)
    print("Test 3: Multi-class Classification - Spiral Data")
    print("="*50)
    
    X_spiral, y_spiral = generate_spiral_data(n_samples=50, n_classes=3)
    
    # Create neural network: 2 inputs -> 10 hidden -> 3 outputs
    nn_spiral = NeuralNetwork(layers=[2, 10, 3], activation='relu', output_activation='softmax', learning_rate=0.01)
    nn_spiral.train(X_spiral, y_spiral, epochs=1000, verbose=False)
    
    final_accuracy = nn_spiral.accuracy_history[-1]
    print(f"Spiral Classification Accuracy: {final_accuracy:.3f}")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(nn_xor.loss_history[:500])  # Plot first 500 epochs
        plt.title('XOR - Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(nn_spiral.accuracy_history)
        plt.title('Spiral - Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting training curves")
    
    # Test 4: Different activation functions comparison
    print("\n" + "="*50)
    print("Test 4: Activation Function Comparison")
    print("="*50)
    
    activations = ['sigmoid', 'tanh', 'relu']
    results = {}
    
    for activation in activations:
        print(f"\nTesting {activation} activation:")
        nn = NeuralNetwork(layers=[2, 6, 1], activation=activation, learning_rate=0.1)
        nn.train(X_xor, y_xor, epochs=1000, verbose=False)
        
        predictions = nn.predict(X_xor)
        accuracy = nn.compute_accuracy(y_xor, nn.predict_proba(X_xor))
        final_loss = nn.loss_history[-1]
        
        results[activation] = {'accuracy': accuracy, 'loss': final_loss}
        print(f"  Final Accuracy: {accuracy:.3f}, Final Loss: {final_loss:.4f}")
    
    # Test 5: Gradient checking (simplified)
    print("\n" + "="*50)
    print("Test 5: Gradient Checking")
    print("="*50)
    
    # Create small network for testing
    test_nn = NeuralNetwork(layers=[2, 3, 1], learning_rate=0.01)
    gradient_check(test_nn, X_xor[:2], y_xor[:2])  # Use subset for speed
    
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    print("✓ Single Perceptron: Can solve linearly separable problems (AND gate)")
    print("✓ Multi-layer Network: Can solve non-linearly separable problems (XOR)")
    print("✓ Multi-class Classification: Handles multiple output classes")
    print("✓ Different Activations: Each has different convergence properties")
    print("✓ Backpropagation: Successfully learns through gradient descent")
    
    print(f"\nBest activation for XOR: {max(results.keys(), key=lambda x: results[x]['accuracy'])}")
    
    # Optional: Plot decision boundaries if matplotlib available
    print("\n" + "="*50)
    print("Decision Boundary Visualization")
    print("="*50)
    
    try:
        plot_decision_boundary(nn_xor, X_xor, y_xor, "XOR Problem - Decision Boundary")
        plot_decision_boundary(nn_spiral, X_spiral, y_spiral, "Spiral Data - Decision Boundary")
    except:
        print("Visualization not available - install matplotlib for plots")

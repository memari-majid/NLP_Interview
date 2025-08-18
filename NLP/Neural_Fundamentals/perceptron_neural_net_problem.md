# Problem: Neural Network from Scratch

Implement basic neural networks from scratch:
1. `Perceptron(input_size: int, learning_rate: float)` - Single perceptron
2. `NeuralNetwork(layers: List[int], activation: str)` - Multi-layer network  
3. `train(X: np.ndarray, y: np.ndarray, epochs: int)` - Training with backpropagation
4. `predict(X: np.ndarray) -> np.ndarray` - Forward pass prediction

Example:
XOR problem: X = [[0,0], [0,1], [1,0], [1,1]], y = [0, 1, 1, 0]
Network: [2, 4, 1] (2 inputs, 4 hidden, 1 output)
Result: Learns XOR function with ~95% accuracy

Requirements:
- Implement forward propagation
- Implement backpropagation with chain rule
- Support multiple activation functions (sigmoid, tanh, ReLU)
- Handle binary and multi-class classification
- Add momentum and learning rate decay

Follow-ups:
- Implement different optimizers (Adam, RMSprop)
- Add regularization (dropout, weight decay)
- Gradient checking for debugging
- Mini-batch training

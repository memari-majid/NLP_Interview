# 30-Day ML Coding Interview Study Plan

## Overview

This comprehensive 30-day study plan is designed to prepare you for machine learning coding interviews at top tech companies. The plan balances theoretical understanding with hands-on implementation, focusing on the most commonly asked topics and problems.

## Study Schedule: 2-3 hours per day

### Week 1: Foundations and Data Preprocessing (Days 1-7)

#### Day 1: Mathematical Foundations
**Time Allocation: 3 hours**
- **Morning (1.5h)**: Linear Algebra Review
  - Vectors, matrices, eigenvalues/eigenvectors
  - Matrix multiplication, transpose, inverse
  - Singular Value Decomposition (SVD)
- **Afternoon (1.5h)**: Statistics and Probability
  - Descriptive statistics, distributions
  - Bayes' theorem, conditional probability
  - Central limit theorem

**Practice:** 
- Implement basic matrix operations from scratch
- Calculate probability distributions
- **Resource**: Khan Academy Linear Algebra

#### Day 2: Data Preprocessing Fundamentals
**Time Allocation: 2.5 hours**
- **Theory (1h)**: Types of missing data, preprocessing pipeline
- **Implementation (1.5h)**: 
  - [Handling Missing Data](../data_preprocessing/handling_missing_data.md)
  - [Problem 001: Data Preprocessing](../practice_problems/easy/problem_001_data_preprocessing.md)

**Practice:**
- Implement multiple imputation strategies
- Build preprocessing pipeline from scratch
- Handle categorical variables

#### Day 3: Feature Engineering and Selection
**Time Allocation: 2.5 hours**
- **Morning (1h)**: Feature scaling, normalization, standardization
- **Afternoon (1.5h)**: Feature selection methods
  - Filter methods (correlation, chi-square)
  - Wrapper methods (recursive feature elimination)
  - Embedded methods (L1 regularization)

**Practice:**
- Implement MinMaxScaler and StandardScaler from scratch
- Build feature selection algorithms
- Compare different scaling techniques

#### Day 4: Exploratory Data Analysis (EDA)
**Time Allocation: 2 hours**
- **Theory (0.5h)**: EDA best practices
- **Implementation (1.5h)**: 
  - Visualization techniques
  - Statistical analysis
  - Pattern detection

**Practice:**
- Create comprehensive EDA pipeline
- Implement outlier detection methods
- Build data profiling tools

#### Day 5: Linear Regression Deep Dive
**Time Allocation: 3 hours**
- **Theory (1h)**: Mathematical derivation, assumptions
- **Implementation (2h)**: 
  - [Linear Regression from Scratch](../algorithms/linear_regression.md)
  - Gradient descent implementation
  - Normal equation method

**Practice:**
- Implement both gradient descent and analytical solutions
- Add regularization (Ridge, Lasso)
- Performance evaluation and visualization

#### Day 6: Logistic Regression
**Time Allocation: 2.5 hours**
- **Theory (1h)**: Sigmoid function, maximum likelihood estimation
- **Implementation (1.5h)**: 
  - Binary and multiclass classification
  - Cross-entropy loss function
  - Regularization techniques

**Practice:**
- Build logistic regression from scratch
- Implement one-vs-all for multiclass
- Compare with sklearn implementation

#### Day 7: Week 1 Review and Assessment
**Time Allocation: 2 hours**
- **Review (1h)**: Consolidate week's learning
- **Assessment (1h)**: Solve 2-3 easy problems
- **Planning**: Identify weak areas for next week

**Practice:**
- Complete all easy preprocessing problems
- Build end-to-end data pipeline
- Self-assessment quiz

---

### Week 2: Core ML Algorithms (Days 8-14)

#### Day 8: Decision Trees
**Time Allocation: 3 hours**
- **Theory (1h)**: 
  - Information gain, Gini impurity, entropy
  - Tree construction algorithms
  - Pruning techniques
- **Implementation (2h)**: Build decision tree from scratch

**Practice:**
- Implement ID3/C4.5 algorithm
- Add pruning mechanisms
- Visualize decision boundaries

#### Day 9: Ensemble Methods - Random Forest
**Time Allocation: 3 hours**
- **Theory (1h)**: Bagging, bootstrap sampling
- **Implementation (2h)**: 
  - Random forest from scratch
  - Feature importance calculation
  - Out-of-bag error estimation

**Practice:**
- Build random forest classifier and regressor
- Implement feature importance ranking
- Compare with single decision tree

#### Day 10: Support Vector Machines (SVM)
**Time Allocation: 3 hours**
- **Theory (1.5h)**: 
  - Margin maximization, support vectors
  - Kernel trick, different kernels
  - Soft margin and C parameter
- **Implementation (1.5h)**: SVM with different kernels

**Practice:**
- Implement SVM with linear kernel
- Experiment with RBF and polynomial kernels
- Hyperparameter tuning

#### Day 11: K-Means Clustering
**Time Allocation: 2.5 hours**
- **Theory (1h)**: 
  - Centroid-based clustering
  - Elbow method, silhouette analysis
  - Limitations and alternatives
- **Implementation (1.5h)**: [KNN Implementation Problem](../practice_problems/medium/problem_001_implement_knn.md)

**Practice:**
- K-means from scratch
- Implement elbow method
- Handle initialization strategies

#### Day 12: K-Nearest Neighbors (KNN)
**Time Allocation: 2.5 hours**
- **Theory (0.5h)**: Distance metrics, curse of dimensionality
- **Implementation (2h)**: Complete KNN implementation

**Practice:**
- Multiple distance metrics
- Weighted voting
- Optimization for large datasets

#### Day 13: Naive Bayes
**Time Allocation: 2 hours**
- **Theory (1h)**: 
  - Bayes' theorem application
  - Gaussian, Multinomial, Bernoulli variants
  - Laplace smoothing
- **Implementation (1h)**: Naive Bayes from scratch

**Practice:**
- All three Naive Bayes variants
- Text classification example
- Handle zero probabilities

#### Day 14: Week 2 Review and Mid-Point Assessment
**Time Allocation: 3 hours**
- **Review (1h)**: Algorithm comparison and selection
- **Practice (2h)**: Solve medium-difficulty problems
- **Assessment**: Mock interview with algorithms

---

### Week 3: Neural Networks and Deep Learning (Days 15-21)

#### Day 15: Neural Network Fundamentals
**Time Allocation: 3 hours**
- **Theory (1.5h)**: 
  - Perceptron, multilayer networks
  - Activation functions
  - Universal approximation theorem
- **Implementation (1.5h)**: Single-layer perceptron

**Practice:**
- Implement perceptron from scratch
- Test different activation functions
- Visualize decision boundaries

#### Day 16: Backpropagation Algorithm
**Time Allocation: 3.5 hours**
- **Theory (1.5h)**: 
  - Chain rule, computational graphs
  - Forward and backward passes
  - Gradient computation
- **Implementation (2h)**: Multi-layer neural network

**Practice:**
- Full backpropagation implementation
- Manual gradient calculations
- Compare with automatic differentiation

#### Day 17: Optimization and Regularization
**Time Allocation: 3 hours**
- **Theory (1h)**: 
  - SGD, Adam, RMSprop
  - Learning rate scheduling
  - Dropout, batch normalization
- **Implementation (2h)**: Add optimizers to neural network

**Practice:**
- Implement different optimizers
- Add regularization techniques
- Learning rate experimentation

#### Day 18: Convolutional Neural Networks (CNNs)
**Time Allocation: 3 hours**
- **Theory (1.5h)**: 
  - Convolution operation, pooling
  - CNN architecture design
  - Parameter sharing, translation invariance
- **Implementation (1.5h)**: Basic CNN layers

**Practice:**
- Implement convolution and pooling layers
- Build simple CNN for image classification
- Understand parameter calculations

#### Day 19: Recurrent Neural Networks (RNNs)
**Time Allocation: 3 hours**
- **Theory (1h)**: 
  - Sequence modeling, hidden states
  - Vanishing gradient problem
  - LSTM and GRU variants
- **Implementation (2h)**: Basic RNN and LSTM

**Practice:**
- Vanilla RNN implementation
- LSTM cell from scratch
- Sequence prediction tasks

#### Day 20: Advanced Deep Learning Topics
**Time Allocation: 2.5 hours**
- **Theory (1h)**: 
  - Attention mechanism
  - Transfer learning
  - Generative models (GANs basics)
- **Implementation (1.5h)**: Attention mechanism

**Practice:**
- Simple attention implementation
- Transfer learning example
- GAN discriminator/generator

#### Day 21: Week 3 Review and Deep Learning Assessment
**Time Allocation: 2.5 hours**
- **Review (1h)**: Deep learning concepts consolidation
- **Practice (1.5h)**: Neural network problems
- **Planning**: Prepare for system design week

---

### Week 4: System Design and Advanced Topics (Days 22-28)

#### Day 22: ML System Design Fundamentals
**Time Allocation: 3 hours**
- **Theory (2h)**: 
  - ML system architecture
  - Data pipelines, model serving
  - Monitoring and maintenance
- **Case Study (1h)**: Design a recommendation system

**Practice:**
- Design ML system architectures
- Discuss trade-offs and scalability
- Practice system design interviews

#### Day 23: Model Deployment and Production
**Time Allocation: 2.5 hours**
- **Theory (1.5h)**: 
  - Model serving strategies
  - A/B testing for ML
  - Model versioning and rollback
- **Implementation (1h)**: Model serving simulation

**Practice:**
- Design model deployment pipeline
- Implement A/B testing framework
- Monitoring and alerting systems

#### Day 24: Real-time ML Systems
**Time Allocation: 3 hours**
- **Theory (1.5h)**: 
  - Streaming data processing
  - Online learning algorithms
  - Latency and throughput optimization
- **Case Studies (1.5h)**: 
  - Fraud detection system
  - Real-time recommendation engine

**Practice:**
- Design streaming ML pipeline
- Implement online learning algorithm
- Optimize for low latency

#### Day 25: Feature Stores and Data Engineering
**Time Allocation: 2.5 hours**
- **Theory (1h)**: 
  - Feature store architecture
  - Data versioning and lineage
  - ETL vs ELT for ML
- **Implementation (1.5h)**: Feature store design

**Practice:**
- Design feature engineering pipeline
- Implement data validation
- Handle data drift detection

#### Day 26: Model Evaluation and Metrics
**Time Allocation: 3 hours**
- **Theory (1.5h)**: 
  - Evaluation strategies
  - Cross-validation techniques
  - Business metrics vs ML metrics
- **Implementation (1.5h)**: Comprehensive evaluation framework

**Practice:**
- Implement various evaluation metrics
- Design experiments for model comparison
- Statistical significance testing

#### Day 27: Advanced ML Topics
**Time Allocation: 2.5 hours**
- **Theory (1.5h)**: 
  - Reinforcement learning basics
  - Unsupervised learning applications
  - AutoML and hyperparameter optimization
- **Implementation (1h)**: Hyperparameter tuning

**Practice:**
- Grid search and random search
- Bayesian optimization
- AutoML pipeline

#### Day 28: System Design Practice
**Time Allocation: 3 hours**
- **Mock Interviews (3h)**: 
  - Complete system design problems
  - Recommendation systems
  - Search and ranking systems
  - Fraud detection systems

**Practice:**
- Time-constrained system design
- Communication and presentation
- Handling follow-up questions

---

### Final Days: Review and Mock Interviews (Days 29-30)

#### Day 29: Comprehensive Review
**Time Allocation: 4 hours**
- **Morning (2h)**: Review all algorithms and implementations
- **Afternoon (2h)**: Solve mixed difficulty problems

**Focus Areas:**
- Algorithm comparison and selection
- Implementation details and optimizations
- Common pitfalls and debugging

#### Day 30: Final Mock Interviews
**Time Allocation: 4 hours**
- **Coding Interview (2h)**: Algorithm implementation
- **System Design Interview (2h)**: ML system architecture

**Final Preparation:**
- Practice whiteboard coding
- Prepare for behavioral questions
- Review company-specific problems

---

## Daily Study Structure

### Morning Routine (30 minutes)
1. **Review previous day's concepts** (10 min)
2. **Quick algorithm quiz** (10 min)
3. **Read one research paper abstract** (10 min)

### Main Study Session (2-3 hours)
1. **Theory Learning** (30-60 min)
2. **Implementation Practice** (60-90 min)
3. **Problem Solving** (30-60 min)

### Evening Review (20 minutes)
1. **Summarize key learnings** (10 min)
2. **Plan next day's focus** (10 min)

---

## Weekly Assessments

### Week 1 Assessment: Fundamentals
- [ ] Implement linear/logistic regression from scratch
- [ ] Build complete data preprocessing pipeline
- [ ] Solve 5 easy problems correctly

### Week 2 Assessment: Core Algorithms
- [ ] Implement 3 algorithms from scratch
- [ ] Compare algorithm performance on datasets
- [ ] Solve 3 medium problems correctly

### Week 3 Assessment: Deep Learning
- [ ] Build neural network with backpropagation
- [ ] Implement CNN or RNN variant
- [ ] Explain optimization techniques

### Week 4 Assessment: System Design
- [ ] Design 2 complete ML systems
- [ ] Discuss trade-offs and scaling
- [ ] Handle system design interview questions

---

## Resources and Tools

### Essential Books
- "Hands-On Machine Learning" by Aurélien Géron
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Online Resources
- **Coursera**: Andrew Ng's ML Course
- **Fast.ai**: Practical Deep Learning
- **Kaggle Learn**: Free micro-courses
- **Papers With Code**: Latest research implementations

### Practice Platforms
- **LeetCode**: Coding problems with ML focus
- **HackerRank**: ML challenges
- **Kaggle**: Real-world datasets and competitions
- **GeeksforGeeks**: Algorithm implementations

### Implementation Tools
- **Python Libraries**: NumPy, Pandas, Scikit-learn
- **Deep Learning**: TensorFlow, PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Jupyter Notebooks**: Interactive development

---

## Tips for Success

### Coding Best Practices
1. **Always start with brute force**, then optimize
2. **Explain your thought process** out loud
3. **Handle edge cases** and validate inputs
4. **Test with examples** and verify correctness
5. **Discuss time/space complexity**

### Interview Strategies
1. **Ask clarifying questions** before coding
2. **Think out loud** during implementation
3. **Start with simple cases**, build complexity
4. **Test your solution** with examples
5. **Be ready to optimize** and discuss alternatives

### Common Pitfalls to Avoid
- Not understanding the problem fully
- Jumping to complex solutions immediately
- Forgetting to handle edge cases
- Poor variable naming and code structure
- Not testing the solution thoroughly

---

## Company-Specific Preparation

### Google/Meta Focus Areas
- Large-scale system design
- Advanced algorithms and optimization
- TensorFlow/PyTorch proficiency
- Distributed computing concepts

### Amazon/Microsoft Focus Areas
- Business metric optimization
- Cost-effective solutions
- Cloud platform integration
- Practical ML applications

### Startup Focus Areas
- End-to-end ML pipeline
- Resource constraints handling
- Rapid prototyping
- Product impact measurement

---

## Progress Tracking

### Daily Checklist
- [ ] Theoretical concepts reviewed
- [ ] Implementation completed
- [ ] Problems solved correctly
- [ ] Key learnings documented
- [ ] Next day planned

### Weekly Goals
- [ ] All planned topics covered
- [ ] Implementation projects completed
- [ ] Assessment problems solved
- [ ] Weak areas identified
- [ ] Next week prepared

### Final Readiness Checklist
- [ ] Can implement 10+ algorithms from scratch
- [ ] Comfortable with neural network fundamentals
- [ ] Can design ML systems end-to-end
- [ ] Familiar with production ML challenges
- [ ] Ready for behavioral interviews

---

**Remember**: Consistency is key! Better to study 2 hours daily than to cram 14 hours on weekends. Focus on understanding concepts deeply rather than memorizing implementations.

**Good luck with your ML interview preparation!** 🚀

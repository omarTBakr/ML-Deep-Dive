 # My Machine Learning Algorithm Deep Dives

## Project Goal

Welcome! This repository is my coding playground for deeply understanding machine learning algorithms. It's not just about implementing algorithms "from scratch," but also about exploring their:

* **Mathematical Underpinnings:**  Deriving equations, analyzing complexity, and understanding why they work.
* **Performance Characteristics:**  Comparing different implementations (iterative vs. vectorized), exploring optimization techniques, and analyzing time and space complexity.
* **Alternative Solutions:**  Contrasting various approaches to solve the same problem (e.g., Normal Equation vs. Gradient Descent) and understanding their trade-offs. 

## Target Audience

This repository is for you if you're:

* An aspiring data scientist or ML enthusiast who wants to go beyond black-box implementations and gain a profound understanding of ML algorithms.
* Anyone looking to solidify their ML foundations by implementing algorithms, analyzing their performance, and comparing different solution strategies.

## Prerequisites

* **Solid Python Skills:** Be comfortable with Python syntax, data structures, control flow, functions, and basic file operations.
* **Linear Algebra:**  Understanding vectors, matrices, and operations like dot products and matrix multiplication will be very beneficial.
* **Analytical Mindset:** Be ready to dive into mathematical derivations, analyze code complexity, and compare different approaches critically.

## Repository Structure

Each algorithm in this repository includes:

* **Detailed README.md:** Explanation of the algorithm, its applications, mathematical background, implementation details, complexity analysis, and comparisons between different solutions.
* **Python Code:**  Well-commented code showcasing implementations (often multiple for comparison), with a focus on clarity and educational value.
* **Data & Examples:**  Datasets used for testing and demonstration, along with examples of using the implemented algorithms.

## Current Deep Dives

* **Linear Regression:** [here](https://github.com/omarTBakr/Educational-ML-/tree/main/linear%20regression#readme)
    - **Gradient Descent Implementation:** 
       - Iterative approach for step-by-step learning.
       - Vectorized implementation using NumPy for efficiency.
       - Time Complexity Analysis: O(n * k * d) 
    - **Hyperparameter Tuning:**  Finding optimal parameters for improved performance.
    - **Data Preprocessing:**  Min-Max Scaling, Standardisation, and their impact.
    - **Normal Equation:**  A direct solution method as an alternative to Gradient Descent.
       - Implementation and explanation of its derivation.
       - Time Complexity Analysis:  O(d^3)
       - Comparison with Gradient Descent based on dataset characteristics.
     
*  **Polynomial Regression:** [here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/Polynomial%20regression)
    -  **Extends linear regression:** Models non-linear input-output relationships by transforming features with polynomial functions (x², x³, etc.).
    -  **Higher-dimensional space:**  Projects data into a higher dimension where linear relationships may exist.
    -  **Model fitting:**  Uses standard linear regression techniques on the transformed features.
    -  **Finding the optimal degree:**
       * **Experiment:** Test different polynomial degrees and observe their impact on model fit.
       * **Error analysis (MSE):** Identify the degree where error reduction plateaus, indicating a balance between bias and variance.
       * **BIC:** Statistically penalises model complexity to guide optimal degree selection.
     
*  **hands-on project diving into regression and regularisation techniques in Python using scikit-learn**[here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/RegressorsAndRegularization)


   It guides you through:
   - **Exploring polynomial regression** by varying degrees and analysing its impact on error, more on that [here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/Polynomial%20regression).
   - **Implementing custom functions for monomial feature** creation and error visualisation.
   - **Applying Ridge and Lasso regularization** to optimize model performance and perform feature selection.
   - **Building a custom regularized linear regression class with NumPy**.
   - **Constructing a machine learning pipeline with data preprocessing, polynomial feature engineering, and Ridge regression using GridSearchCV for hyperparameter tuning**.

* **Neural Networks (From First Principles):** [here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/NeuralNetwork)
    - **Building Blocks:** Constructing `Neuron`, `NeuralLayer`, and `NeuralNetwork` classes from the ground up to demystify their interactions.
    - **Core Algorithms:** Step-by-step implementation of feedforward and backpropagation with minimal library abstraction, focusing on the underlying mechanics.
    - **Practical Context:** Includes a Scikit-learn `MLPRegressor` example with `GridSearchCV` for a complete learning perspective.
 
*  **Neural Network Vectorize Implemntations)** [here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/NeuralNetworkVectorized)

   
    - **Implementing a network form scratch in a vectorized manner**
    - **Flexible architecture:** Easily adjust the number of hidden units and activation functions.
    - **practice on MNIST data set**
    - **Softmax, Cross-Entropy, and KL-Divergence , with some theoritical and practical questions**
    - **Explores the mathematical relationship and similarities between cross-entropy and KL-divergence in the context of neural network training.**
   
    - der_softmax.py for softmax and its derivatives [here](https://github.com/omarTBakr/ML-Deep-Dive/blob/main/NeuralNetworkVectorized/der_softmax.py)
    - multiclass_classification_theoritical_quesetions.pdf for theory and derivations  [here](https://github.com/omarTBakr/ML-Deep-Dive/blob/main/NeuralNetworkVectorized/multiclass_classification_theoritical_quesetions.pdf)
      
* **KMeans implementation from scratch using numpy** [here](https://github.com/omarTBakr/ML-Deep-Dive/tree/main/Kmeans)

  - You will find
  - **clear implementation using Python**
  - **How to initialize k ?**
  - **Which number of clusters to use?**
  - **comparing my implementation against `sklearn.cluster.KMeans`**
 
    
The project aims to provide practical experience with these essential machine learning concepts and tools. 

## Future Deep Dives (Roadmap)

* **Support Vector Machines (SVM):**  Different kernels, optimization techniques.
 
* **Clustering Algorithms:**  K-Means, DBSCAN, hierarchical clustering.

## Contributing

I welcome contributions! If you'd like to suggest improvements, point out errors, or add an algorithm deep dive, feel free to open an issue or submit a pull request! 

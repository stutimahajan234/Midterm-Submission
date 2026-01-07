# Midterm-Submission
This repository contains all the coding work completed up to the midterm. The work is implemented using Python and Jupyter Notebooks.

## Repository Structure

## Assignment 1

### Logic Overview

Each code block performs a standalone task:

1. Input and Output Handling
   - Takes numerical input from the user
   - Demonstrates basic input validation using "try-except"

2. Array/List Operations
   - Accepts multiple inputs from the user
   - Stores values in a list
   - Performs basic operations on the list

3. Matrix Operations
   - Accepts a square matrix as input
   - Extracts diagonal elements
   - Computes the exponential of diagonal elements
   - Calculates the sum of exponentials

### Key Concepts Used
- Loops (`for`, `while`)
- Conditional handling
- Exception handling
- Lists and matrices
- Mathematical operations using `math` library


## Assignment 2 

## Implements a Logistic Regression model from scratch


## Function Descriptions

### `load_data(filename)`
- Loads dataset from a file
- Separates features and labels
- Returns input matrix `X` and output vector `y`


### `normalize(X)`
- Normalizes input features
- Ensures all features are on a comparable scale


### `add_bias(X)`
- Adds a bias term to the input matrix
- Required for logistic regression formulation


### `sigmoid(z)`
- Implements the sigmoid activation function  
- Converts linear output into probabilities between 0 and 1


### `compute_loss(y, y_pred)`
- Computes the logistic loss 
- Measures how well predictions match true labels


### `gradient_descent(X, y, learning_rate=0.01, epochs=2000)`
- Core training function
- Initializes weights
- Iteratively updates weights using gradient descent
- Minimizes loss over multiple epochs
- Returns trained weights


### `predict(X, weights, threshold=0.5)`
- Generates predictions using trained weights
- Applies sigmoid function
- Converts probabilities to class labels using a threshold


### `accuracy(y_true, y_pred)`
- Compares predicted labels with true labels
- Computes classification accuracy


### `main()`
- Acts as the driver function
- Calls all other functions in sequence:
  - Data loading
  - Normalization
  - Training
  - Prediction
  - Accuracy evaluation

## Tools & Libraries Used
- Python
- NumPy
- Math
- Jupyter Notebook

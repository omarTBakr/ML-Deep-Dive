
# Multiclass Classification Neural Network (Educational)

This project demonstrates a simple, fully vectorized neural network for multiclass classification, implemented from scratch in Python using only NumPy. The code is designed for educational purposes, focusing on clarity and step-by-step understanding of neural network mechanics.

## Features
- Vectorized implementation (no explicit loops over samples in forward/backward passes)
- Flexible architecture: choose number of hidden units and activation functions
- Trains and evaluates on the MNIST handwritten digit dataset
- Clear comments and docstrings for beginners


## Files
- `NNVectorized.py`: Main neural network implementation and training script
- `loaddata.py`: Utility for downloading and loading the MNIST dataset using torchvision
- `der_softmax.py`: Code for softmax function and its derivatives (Jacobian, both iterative and vectorized)
- `multiclass_classification_theoritical_quesetions.pdf`: PDF with theoretical questions and answers on multiclass classification, softmax, cross-entropy, and KL-divergence (including their similarities and differences)
- `requirements.txt`: Python dependencies
- `05 Homework 1 - Multiclass Classifier.pdf`: Homework/assignment description (if provided)
- `input_output/`: Example input/output numpy arrays (optional, for further exercises)

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python NNVectorized.py
   ```
   The script will download MNIST if needed, train the network, and print training/validation loss and accuracy every 10 epochs.


## Exploring Derivatives & Theory
- See `der_softmax.py` for code to compute the softmax function and its Jacobian (derivative), both with loops and vectorized.
- The PDF `multiclass_classification_theoritical_quesetions.pdf` contains detailed explanations and derivations for softmax, cross-entropy, and KL-divergence, and discusses their relationships.

## Customization
- Change the number of hidden units or activation functions in `NNVectorized.py` to experiment with different architectures.
- The code is easy to extend for more layers or other datasets.

## Credits
Made by GPT with the assistance of the author. For educational use only.

# Neural Network From Scratch (NumPy)

This project is a simple implementation of a one-hidden-layer neural
network built only with NumPy.\
I made this to understand how forward pass, backpropagation and weight
updates actually work under the hood.

## The repository includes:

-   ReLU and Sigmoid activation functions\
-   Manual weight initialization\
-   Forward propagation\
-   Binary cross-entropy loss\
-   Backpropagation\
-   A small training loop using synthetic data (just to show that it
    runs)

## This repo does not include:

-   The real dataset I used\
-   Preprocessing steps\
-   Scaling, splitting, or evaluation metrics

Those were part of my own experimentation and kept separate.\
Here I'm only sharing the model logic so the core neural network math is
easy to follow.

## How to Run

Clone the repo and run:

    python train.py

This will train on dummy randomly generated data.\
The purpose is just to demonstrate how the functions work together.

## Using the Model With a Real Dataset

The code in this repository uses synthetic data only, just to
demonstrate the forward pass, loss calculation and backpropagation.\
If you want to train the model on a real dataset, you can follow this
general workflow:

1.  Load your dataset using pandas or NumPy.\
2.  Select the input features you want to use and prepare your target
    variable.\
3.  Preprocess the data as needed (cleaning, scaling, handling missing
    values).\
4.  Convert everything into NumPy arrays:\
    `X_train, y_train = ...`\
5.  Pass your NumPy arrays into the existing training loop in
    `train.py`.\
6.  Run the script to train the network on your data.

**Example (conceptual only):**\
`X_train_np = your_feature_matrix`\
`y_train_np = your_labels.reshape(-1, 1)`

The rest of the training loop remains exactly the same.

## Why I Built This

This was mainly a learning exercise to understand:

-   How gradients flow\
-   How loss changes with each update\
-   How shapes and matrix operations interact\
-   What actually happens inside a basic neural network

It helped me move beyond just calling `.fit()` from a library and
actually see what's happening step by step.

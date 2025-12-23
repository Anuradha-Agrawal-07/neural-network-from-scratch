# Neural Network From Scratch (NumPy)

This project is a simple implementation of a one-hidden-layer neural
network built only with NumPy.\
I made this to understand how forward pass, backpropagation and weight
updates actually work under the hood.

The repository includes: - relu and sigmoid activation functions -
manual weight initialization - forward propagation - binary
cross-entropy loss - backpropagation - a small training loop using
synthetic data (just to show that it runs)

This repo does **not** include: - the real dataset I used -
preprocessing steps - scaling, splitting or evaluation metrics

Those were part of my own experimentation and kept separate.\
Here I'm only sharing the model logic so the core neural network math is
easy to follow.

## How to Run

Clone the repo and run:

    python train.py

This will train on dummy randomly generated data.\
The purpose is just to demonstrate how the functions work together.

## Why I Built This

This was mainly a learning exercise to understand: - how gradients
flow - how loss changes with each update - how shapes and matrix
operations interact - what actually happens inside a basic neural
network

It helped me move beyond just calling `.fit()` from a library and
actually see what's happening step by step.

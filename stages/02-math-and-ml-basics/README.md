# Stage 02 — Math and ML Basics

This stage teaches the maths an AI engineer actually uses, and nothing more. No proofs, no measure theory: vectors, derivatives, probability, and the training loop they add up to. It sits second because every later stage — transformers, fine-tuning, evaluation — assumes you can read a matrix shape and explain what a gradient is.

**Time:** roughly 3-4 weeks part-time · **You need first:** Stage 01, especially [Core Algorithms](../01-programming-basics/04-algorithms.md) · **You leave with:** a NumPy implementation of gradient descent that trains a small model from scratch

## What you will be able to do

- Read a tensor shape like `[batch, seq, dim]` and predict what any operation does to it.
- Compute a derivative by hand, and apply the chain rule through a two-layer network.
- Explain overfitting, and choose a train/validation/test split that tells you the truth.
- Pick a loss function that matches the task instead of copying the default.
- Implement forward and backward passes for a small network in plain NumPy.
- Train it with gradient descent and prove the loss goes down for the right reason.

## Topics

| # | Topic | What it gives you |
|---|-------|-------------------|
| 1 | [Linear Algebra](01-linear-algebra.md) | Vectors, matrices, dot products, shapes as meaning |
| 2 | [Calculus](02-calculus.md) | Derivatives, partials, the chain rule, gradients |
| 3 | [Probability and Bayes](03-probability-bayes.md) | Distributions, conditional probability, updating beliefs |
| 4 | [ML Basics](04-ml-basics.md) | Features, labels, splits, overfitting, baselines |
| 5 | [Loss Functions](05-loss-functions.md) | MSE, cross-entropy, why the choice changes learning |
| 6 | [Backpropagation and Gradients](06-backprop-gradients.md) | Credit assignment by hand through a network |
| 7 | [Ship Gradient Descent](07-ship-gradient-descent.md) | A from-scratch NumPy trainer, no PyTorch |

## The shipping project

[07-ship-gradient-descent.md](07-ship-gradient-descent.md) — you build `micrograd-numpy`, a small library that fits a two-layer network to a toy dataset using only NumPy: manual forward pass, manual backward pass, learning-rate schedule, and a training curve you plot. You prove it works by matching your gradients against a numerical check and showing the loss falling to a known floor.

## How to know you are done

- You can write the shape of every intermediate in a two-layer network without running the code.
- Your numerical gradient check agrees with your analytic gradients to a small tolerance.
- Your trainer beats a mean-prediction baseline on held-out data, and you can say why.
- You can explain cross-entropy to someone else using one sentence about surprise.

---
[← Back to roadmap](../../README.md) · [Next stage: Data Engineering](../03-data-engineering/README.md)

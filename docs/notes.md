# Notes about deep learning

This is for remembering some easy-to-forget stuffs in deep learning, helping you
quickly re-collect them.

[TOC]

## Builtin datasets

-   [MNIST](https://en.wikipedia.org/wiki/MNIST_database) containing images and
    digit classes for image of digits classification.
-   [IMDb](https://imdb.com/) containing ??? for reviews of films.

## Activation function

-   Less useful activation functions like sigmoid and linear function can be
    applied to the **output layer**.
-   Softmax is used to transform ???

## FAQ

### What is Manifold?

I don't know : )

### Optimization, Generalization and Overfitting

_Optimization_ is the process of fitting the model parameters to the input data
while avoiding overfitting. _Generalization_ is the ability of a model to
perform well on new, unseen data that it has not encountered during training.

Overfitting is **one of** the main challenges that can hinder generalization,
and optimization can help to mitigate overfitting by finding a set of parameters
that can accurately map inputs to outputs while avoiding overfitting.

### How to achieve generalization?

By using regularization, data augmentation, early stopping, cross-validation,
ensemble methods and model selection.

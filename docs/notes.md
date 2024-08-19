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

## Matrix weights

### Convolutional layer

For a convolutional layer, for each output feature, the shape of its weight is
`(kernel_size, kernel_size, num_input_features)`.  And the shape the bias is
`(1,)` (which is pretty strange in my opinion, Llama's comments[^1]). Therefore
there are `num_output_features` weights and biases matrices totally (total
number of parameters can be calculated through
`num_output_features * (num_input_features * kernel_size * kernel_size + 1)`.

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

[^1]: In a convolutional layer, the bias term is typically not dependent on the
input channels. This is because the bias term is added to the output of the
convolution operation, which is a sum of the weighted inputs from all input
channels.<br>
Think of it this way: the weights are used to compute the weighted sum of the
input values from all input channels, and the bias term is added to this sum.
Since the bias term is not dependent on the specific input channels, it's not
necessary to have separate bias terms for each input channel.<br>
In other words, the bias term is applied to the output of the convolution
operation, which is a single value that represents the sum of the weighted
inputs from all input channels. This single bias term is sufficient to shift the
output value, regardless of the number of input channels.<br>
This is a common design choice in CNNs, and it's what allows the network to be
efficient and scalable. Having separate bias terms for each input channel would
increase the number of parameters in the network, which can lead to overfitting
and other issues.

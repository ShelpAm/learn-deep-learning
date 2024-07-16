import config

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

DEBUG = True


def make_random_data():
    num_samples_per_class = 10
    cov = [[1, 0], [0, 1]]
    negative_samples = np.random.multivariate_normal(
        mean=[0, 10], cov=cov, size=num_samples_per_class
    )
    positive_samples = np.random.multivariate_normal(
        mean=[10, 0], cov=cov, size=num_samples_per_class
    )

    inputs = np.vstack([negative_samples, positive_samples]).astype(np.float32)
    targets = np.vstack(
        [
            np.zeros((num_samples_per_class, 1), dtype="float32"),
            np.ones((num_samples_per_class, 1), dtype="float32"),
        ]
    )

    return inputs, targets


def logistic_regression_loss(targets, predictions):
    # per_sample_losses = tf.square(targets - predictions)
    per_sample_losses = -(
        targets * tf.math.log(predictions + 1e-10)
        + (1 - targets) * tf.math.log((1 - predictions + 1e-10))
    )
    return tf.reduce_mean(per_sample_losses)


class Model:
    def __init__(self, input_dim, output_dim):
        self.W = tf.Variable(
            initial_value=tf.random.uniform(shape=(input_dim, output_dim))
        )
        self.b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

    def predict(self, inputs, activation):
        result = tf.matmul(inputs, self.W) + self.b
        return activation(result)

    def fit(self, inputs, targets, num_iterations, learning_rate):
        period = num_iterations // 5
        if DEBUG:
            fig, axs = plt.subplots(2, 5)
            axs[0, 0].scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])

        for step in range(num_iterations):
            if DEBUG:
                if step % period == 0:
                    pred = self.predict(inputs, tf.math.sigmoid)
                    axs[1, step // period].scatter(
                        inputs[:, 0], inputs[:, 1], c=pred[:, 0] > 0.5
                    )
                    axs[1, step // period].set_title(f"{step}-th iteration")
            loss = self.training_step(inputs, targets, learning_rate)
            if step % period == 0:
                print(f"Loss at step {step}: {loss:.4f}")
        plt.show()

    def training_step(self, inputs, targets, learning_rate):
        with tf.GradientTape() as tape:
            predictions = self.predict(inputs, tf.math.sigmoid)
            loss = logistic_regression_loss(predictions, targets)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, (self.W, self.b))
        self.W.assign_sub(grad_loss_wrt_W * learning_rate)
        self.b.assign_sub(grad_loss_wrt_b * learning_rate)
        return loss


def run():
    inputs, targets = make_random_data()

    input_dim = 2
    output_dim = 1

    # Initialize variables
    model = Model(input_dim, output_dim)

    model.fit(inputs, targets, 100, learning_rate=0.1)

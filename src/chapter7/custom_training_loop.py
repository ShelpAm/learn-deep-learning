from keras import losses, metrics, optimizers
import tensorflow as tf
from src.utils import make_mnist_dataset, make_mnist_model


def main():
    loss_fn = losses.SparseCategoricalCrossentropy()
    optimizer = optimizers.RMSprop()
    metrics_list = [metrics.SparseCategoricalAccuracy()]
    loss_tracking_metric = metrics.Mean()

    model = make_mnist_model()
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics_list)

    @tf.function
    def training_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(targets, predictions)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        history = {}
        for metric in metrics_list:
            metric.update_state(targets, predictions)
            history[metric.name] = metric.result()
        loss_tracking_metric.update_state(loss)
        history["loss"] = loss_tracking_metric.result()
        return history

    def test_step(inputs, targets):
        predictions = model(inputs, training=True)
        loss = loss_fn(targets, predictions)

        history = {}
        for metric in metrics_list:
            metric.update_state(targets, predictions)
            history["val_" + metric.name] = metric.result()
        loss_tracking_metric.update_state(loss)
        history["val_loss"] = loss_tracking_metric.result()
        return history

    def reset_metrics():
        for metric in metrics_list:
            metric.reset_state()
        loss_tracking_metric.reset_state()

    def training_loop():
        train_dataset, _ = make_mnist_dataset()
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)
        train_dataset = train_dataset.batch(32)
        epochs = 3
        for epoch in range(epochs):
            reset_metrics()
            history = None
            for train_x, train_y in train_dataset:
                history = training_step(train_x, train_y)
            print(f"Result at the end of epoch {epoch}")
            print(history)

    training_loop()

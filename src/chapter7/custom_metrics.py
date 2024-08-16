from os import path
from typing import cast
from keras import Model, callbacks, losses, metrics, models, optimizers, initializers
from tensorflow import (
    one_hot,
    sqrt,
    reduce_sum,
    shape,
    square,
)

from src.utils import make_mnist_dataset, make_mnist_model


# It can do a RMSE on a vector, whereas keras.metrics.RootMeanSquaredError can not.
class Root_mean_squared_error(metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name, kwargs)
        self._mse_sum = self.add_weight(
            initializer=initializers.Zeros(),
            name="mse_sum",
        )
        self._total_samples = self.add_weight(
            initializer=initializers.Zeros(),
            # dtype=int32, # Leads to run error. Don't use it.
            name="total_samples",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = one_hot(y_true, depth=shape(y_pred)[1])  # Convert to one hot.
        mse = reduce_sum(square(y_true - y_pred))
        self._mse_sum.assign_add(mse)
        num_samples = shape(y_pred)[0]
        self._total_samples.assign_add(num_samples)

    def reset_state(self):
        self._mse_sum.assign(0.0)
        self._total_samples.assign(0.0)

    def result(self):
        return sqrt(self._mse_sum / self._total_samples)


def main():
    model_save_path = "bin/custom-metrics.keras"

    (train_x, train_y), (test_x, test_y) = make_mnist_dataset()
    if path.exists(model_save_path):
        model = cast(Model, models.load_model(model_save_path))
    else:
        model = make_mnist_model()
        model.compile(
            optimizer=optimizers.RMSprop(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy(), Root_mean_squared_error()],
        )
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor="val_sparse_categorical_accuracy",
                patience=2,
                mode="max",
            ),
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
            ),
        ]
        model.fit(
            train_x,
            train_y,
            epochs=30,
            validation_split=0.2,
            callbacks=callbacks_list,
        )
    test_metrics = model.evaluate(test_x, test_y)
    print(f"Test metrics: {test_metrics}")

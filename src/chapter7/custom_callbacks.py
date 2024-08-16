from keras import callbacks, losses, metrics, optimizers
from matplotlib import pyplot as plt

from src.utils import make_mnist_dataset, make_mnist_model


class Loss_logger(callbacks.Callback):
    def on_train_begin(self, logs=None):
        self._per_batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self._per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        plt.clf()
        plt.plot(
            range(len(self._per_batch_losses)),
            self._per_batch_losses,
            label="Training loss for each batch",
        )
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"bin/plot_at_epoch_{epoch}")
        # plt.show()
        self._per_batch_losses.clear()


def main():
    (train_x, train_y), _ = make_mnist_dataset()
    model = make_mnist_model()
    model.compile(
        optimizer=optimizers.RMSprop(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )
    model.fit(train_x, train_y, epochs=3, callbacks=[Loss_logger()])

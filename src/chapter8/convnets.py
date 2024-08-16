from os import makedirs, path
from pathlib import Path
from shutil import copyfile
from typing import cast

from keras import (
    Model,
    Sequential,
    activations,
    callbacks,
    layers,
    losses,
    metrics,
    models,
    optimizers,
    utils,
)
from matplotlib.pyplot import axis, figure, imshow, legend, plot, show, subplot, title

from src.utils import make_mnist_dataset, make_mnist_model


def digits_classification():
    (train_images, train_labels), (test_x, test_y) = make_mnist_dataset()
    model = make_mnist_model()

    model.compile(
        optimizer=optimizers.RMSprop(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print(f"test loss: {test_loss}, test accuracy: {test_accuracy}")

    model.summary()


def dog_cat_classification():
    def initialize_basedir(basedir):
        original_dir = Path("res/dogs-vs-cats/train")

        def make_subset(subset_name, start_index, end_index):
            if path.exists(basedir / subset_name):
                return
            for category in ["cat", "dog"]:
                dir = basedir / subset_name / category
                makedirs(dir)
                filenames = [
                    f"{category}.{i}.jpg" for i in range(start_index, end_index)
                ]
                for filename in filenames:
                    copyfile(original_dir / filename, dir / filename)

        train_size, validation_size, test_size = 1000, 500, 1000
        make_subset("train", start_index=0, end_index=train_size)
        make_subset(
            "validation",
            start_index=train_size,
            end_index=train_size + validation_size,
        )
        make_subset(
            "test",
            start_index=train_size + validation_size,
            end_index=train_size + validation_size + test_size,
        )

    def prepare_datasets(basedir):
        image_size = (180, 180)
        # num_channels = 3

        train_dataset = utils.image_dataset_from_directory(
            basedir / "train",
            image_size=image_size,
            batch_size=32,
        )
        validation_dataset = utils.image_dataset_from_directory(
            basedir / "validation",
            image_size=image_size,
            batch_size=32,
        )
        test_dataset = utils.image_dataset_from_directory(
            basedir / "test",
            image_size=image_size,
            batch_size=32,
        )
        return train_dataset, validation_dataset, test_dataset

    def make_compiled_model():
        data_augmentor = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2),
        ])

        # Display some augmented images.
        # figure()
        # for images, _ in train_dataset.take(1):
        #     for i in range(9):
        #         augmenteds = data_augmentor(images)
        #         ax = subplot(3, 3, i + 1)
        #         imshow(augmenteds[0].numpy().astype("uint8"))
        #         axis("off")

        inputs = layers.Input(shape=(180, 180, 3))

        x = data_augmentor(inputs)
        x = layers.Rescaling(1.0 / 255)(x)
        for i in range(5, 9):
            x = layers.Convolution2D(
                filters=2**i, kernel_size=3, activation=activations.relu
            )(x)
            x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Convolution2D(
            filters=2**8, kernel_size=3, activation=activations.relu
        )(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(1, activation=activations.sigmoid)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        model.compile(
            optimizer=optimizers.RMSprop(),
            loss=losses.BinaryCrossentropy(),
            metrics=[metrics.BinaryAccuracy()],
        )
        return model

    def the_model() -> Model:
        if path.exists(model_save_path):
            return cast(Model, models.load_model(model_save_path))

        num_epochs = 100
        model_callbacks = [
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                save_best_only=True,
                monitor="val_loss",
            )
        ]
        model = make_compiled_model()
        history = model.fit(
            train_dataset,
            epochs=num_epochs,
            validation_data=validation_dataset,
            callbacks=model_callbacks,
        )

        epochs = range(1, num_epochs + 1)
        history = history.history
        validation_loss = history["val_loss"]
        validation_accuracy = history["val_binary_accuracy"]
        train_loss = history["loss"]
        train_accuracy = history["binary_accuracy"]

        plot(epochs, train_accuracy, label="Train accuracy")
        plot(epochs, validation_accuracy, label="Validation accuracy")
        title("Train and validation accuracy")
        legend()
        figure()
        plot(epochs, train_loss, label="Train loss")
        plot(epochs, validation_loss, label="Validation loss")
        title("Train and validation loss")
        legend()
        show()

        return model

    # Model settings.
    basedir = Path("res/dogs-vs-cats-small")
    model_save_path = "bin/cat-vs-dog.keras"
    initialize_basedir(basedir)
    train_dataset, validation_dataset, test_dataset = prepare_datasets(basedir)

    model = the_model()
    test_metrics = model.evaluate(test_dataset)
    print(test_metrics)


def main():
    # digits_classification()
    dog_cat_classification()

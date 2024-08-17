from os import path
from keras import (
    datasets,
    layers,
    activations,
    Model,
    Sequential,
)


def make_mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = (
        datasets.mnist.load_data()
    )

    def refine_image(image):
        image = image.reshape((image.shape[0], 28, 28, 1))
        # image = image.astype("float32") / 255
        return image

    train_images = refine_image(train_images)
    test_images = refine_image(test_images)
    return (train_images, train_labels), (test_images, test_labels)


def make_mnist_model():
    data_augmentor = Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

    inputs = layers.Input(shape=(28, 28, 1))

    x = inputs
    # x = data_augmentor(x)
    x = layers.Rescaling(1 / 255)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation=activations.relu)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation=activations.relu)(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation=activations.relu)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(units=10, activation=activations.softmax)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

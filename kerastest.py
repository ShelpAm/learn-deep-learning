import tensorflow as tf
import tensorflow.keras as keras

import mymodel

tf.experimental.numpy.experimental_enable_numpy_behavior()


def get_example_data():
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    def reshape_image(x, y):
        num_samples = x.shape[0]
        return x.reshape(num_samples, -1), y.reshape(num_samples, -1)

    return reshape_image(train_x, train_y), reshape_image(test_x, test_y)


def make_model(sample_shape):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=sample_shape),
            keras.layers.Dense(100, activation="relu"),
            keras.layers.Dense(1000, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="sgd",
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name="acc"),
        ],
    )

    return model


def setup():
    # Load the data and split it between train and test sets
    # (train_x, train_y), (test_x, test_y) = get_example_data()

    n = 5
    params = mymodel.randomized_parameters(1, 1, n)
    train_x, train_y = mymodel.randomized_samples(n=n, m=100, parameters=params)
    test_x, test_y = mymodel.randomized_samples(n=n, m=5, parameters=params)

    train_x = train_x.T
    train_y = train_y.T
    test_x = test_x.T
    test_y = test_y.T

    model = make_model((train_x.shape[1],))
    model.summary()

    model.fit(
        train_x,
        train_y,
        epochs=20,
    )

    print("actual", test_y)
    print("prediction", model.predict(test_x).T)

    # model.save("final_model.keras")
    return model

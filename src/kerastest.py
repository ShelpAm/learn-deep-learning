import keras
import matplotlib.pyplot as plt


def get_example_data():
    # shape of train_x: (, 28, 28)
    (train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

    def flatten_data(x, y):
        num_samples = x.shape[0]
        x = x.astype("float32") / 255
        return x.reshape(num_samples, -1), y
        # .reshape(num_samples, -1)

    return flatten_data(train_x, train_y), flatten_data(test_x, test_y)


def make_model(
    # data_size: int
):
    model = keras.Sequential([
        # keras.layers.Input(shape=(data_size,)),  # Redundant line?
        # May be stricting the input..
        keras.layers.Dense(
            512,
            # kernel_regularizer=keras.regularizers.L2(0.002),
            activation=keras.activations.relu,
        ),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(
            10,
            activation=keras.activations.softmax,
        ),
    ])
    model.compile(
        # "rmsprop",
        optimizer=keras.optimizers.RMSprop(),
        # "sparse_categorical_crossentropy",
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            # "accuracy"
            keras.metrics.SparseCategoricalAccuracy()
        ],
    )
    return model


def setup():
    # Load the data and split it between train and test sets
    (train_x, train_y), (test_x, test_y) = get_example_data()

    model = make_model()

    history = model.fit(
        train_x,
        train_y,
        epochs=16,
        batch_size=64,
        validation_split=0.2,
    )

    # print("Showing training history")
    plt.plot(history.history["loss"], "b-", label="loss")
    plt.plot(
        history.history["sparse_categorical_accuracy"],
        "b--",
        label="sparse_categotical_accuracy",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy & Loss")
    plt.legend()  # Showing legend in the graph.
    plt.show()  # Only configs defined before this take effect.

    # model.save("final_model.keras")
    return model

import tensorflow.keras as keras


def get_example_data():
    (train_images, train_labels), (test_images, test_labels) = (
        keras.datasets.mnist.load_data()
    )

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    # Scale images to the [0, 1] range
    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def make_model():
    # Model parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
        ],
    )

    return model


def setup():
    # Load the data and split it between train and test sets
    (train_images, train_labels), (test_images, test_labels) = get_example_data()

    model = make_model()
    model.summary()

    model.fit(
        train_images,
        train_labels,
        epochs=10,
    )

    # Evaluate the model
    loss, acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    predictions = model.predict(test_images)

    print(predictions)

    model.save("final_model.keras")
    return model

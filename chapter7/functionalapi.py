import keras
import numpy as np


def simple():
    inputs = keras.layers.Input(shape=(3,))  # Symbolic tensor; explicit layer.
    features = keras.layers.Dense(64, keras.activations.relu)(
        inputs
    )  # Again symbolic because it accepted a symbolic tensor.
    outputs = keras.layers.Dense(10, keras.activations.softmax)(
        features
    )  # Again symbolic with the same reason.
    print()
    print("inputs: ", inputs)
    print("features: ", features)
    print("outputs: ", outputs)

    model = keras.Model(inputs, outputs)
    print()
    model.summary()


def multiple_input_and_output():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4

    # Input layers
    title = keras.layers.Input(shape=(vocabulary_size,), name="Title")
    text_body = keras.layers.Input(shape=(vocabulary_size,), name="Text body")
    tags = keras.layers.Input(shape=(num_tags,), name="Tags")

    # Hidden layers
    features = keras.layers.concatenate([title, text_body, tags])
    # Same as `features = layers.Concatenate()([title, text_body, tags])`
    features = keras.layers.Dense(64, activation=keras.activations.relu)(features)

    # Output layers
    priority = keras.layers.Dense(
        1, activation=keras.activations.sigmoid, name="Priority"
    )(features)
    department = keras.layers.Dense(
        num_departments, activation=keras.activations.softmax, name="Department"
    )(features)

    model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss={
            "Priority": keras.losses.MeanSquaredError(),
            "Department": keras.losses.CategoricalCrossentropy(),
        },
        metrics={
            "Priority": keras.metrics.MeanAbsoluteError(),
            "Department": keras.metrics.CategoricalAccuracy(),
        },
    )

    num_samples = 128
    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

    priority_data = np.random.uniform(0, 1, size=(num_samples, 1))
    department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

    model.fit(
        {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
        {"Priority": priority_data, "Department": department_data},
        epochs=5,
    )
    model.evaluate(
        {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
        {"Priority": priority_data, "Department": department_data},
    )


def main():
    simple()
    multiple_input_and_output()

from numpy.random import randint, uniform
from keras import Model, activations, layers, losses, metrics, optimizers, utils


class Customer_ticket_model(Model):
    def __init__(self, num_departments):
        super().__init__()
        self._concat_layer = layers.Concatenate()
        self._mixing_layer = layers.Dense(64, activation=activations.relu)
        self._priority_scorer = layers.Dense(
            1, activation=activations.sigmoid, name="Priority"
        )
        self._department_classifier = layers.Dense(
            num_departments, activation=activations.softmax, name="Department"
        )

    def call(self, inputs):
        title = inputs["Title"]
        text_body = inputs["Text body"]
        tags = inputs["Tags"]

        features = self._concat_layer([title, text_body, tags])
        features = self._mixing_layer(features)

        priority = self._priority_scorer(features)
        department = self._department_classifier(features)

        return {"Priority": priority, "Department": department}


def simple():
    inputs = layers.Input(shape=(3,))  # Symbolic tensor; explicit layer.
    features = layers.Dense(64, activations.relu)(
        inputs
    )  # Again symbolic because it accepted a symbolic tensor.
    outputs = layers.Dense(10, activations.softmax)(
        features
    )  # Again symbolic with the same reason.
    print()
    print("inputs: ", inputs)
    print("features: ", features)
    print("outputs: ", outputs)

    model = Model(inputs, outputs)
    print()
    model.summary()


def multiple_input_and_output():
    vocabulary_size = 10000
    num_tags = 100
    num_departments = 4

    # # Input layers
    # title = layers.Input(shape=(vocabulary_size,), name="Title")
    # text_body = layers.Input(shape=(vocabulary_size,), name="Text body")
    # tags = layers.Input(shape=(num_tags,), name="Tags")
    # # Hidden layers
    # features = layers.concatenate([title, text_body, tags])
    # # Same as `features = layers.Concatenate()([title, text_body, tags])`
    # features = layers.Dense(64, activation=activations.relu)(features)
    #
    # # Output layers
    # priority = layers.Dense(
    #     1, activation=activations.sigmoid, name="Priority"
    # )(features)
    # department = layers.Dense(
    #     num_departments, activation=activations.softmax, name="Department"
    # )(features)
    #
    # model = Model(inputs=[title, text_body, tags], outputs=[priority, department])

    num_samples = 128
    title_data = randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = randint(0, 2, size=(num_samples, num_tags))

    priority_data = uniform(0, 1, size=(num_samples, 1))
    department_data = randint(0, 2, size=(num_samples, num_departments))

    model = Customer_ticket_model(num_departments=num_departments)
    model({
        "Title": title_data,
        "Text body": text_body_data,
        "Tags": tags_data,
    })

    model.compile(
        optimizer=optimizers.RMSprop(),
        loss={
            "Priority": losses.MeanSquaredError(),
            "Department": losses.CategoricalCrossentropy(),
        },
        metrics={
            "Priority": [metrics.MeanAbsoluteError()],
            "Department": [metrics.CategoricalAccuracy()],
        },
    )

    model.fit(
        {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
        {"Priority": priority_data, "Department": department_data},
        epochs=1,
    )
    model.evaluate(
        {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
        {"Priority": priority_data, "Department": department_data},
    )

    # Save topology graph of the model into file
    utils.plot_model(model, "bin/topology.png", show_layer_names=True, show_shapes=True)

    print("\nLayers:")
    for i, layer in enumerate(model.layers):
        print(f"{i}-th layer: ", layer)

    # def incremental_model(model: Model):
    #     # Actually a self-assignment
    #     features = model.layers[4].output
    #     difficulty = layers.Dense(
    #         3, activation=activations.softmax, name="Difficulty"
    #     )(features)
    #     new_difficulty_model = Model([title, text_body, tags], [difficulty])
    #     new_difficulty_model.compile(
    #         optimizer=optimizers.RMSprop(),
    #         loss={
    #             "Difficulty": losses.SparseCategoricalCrossentropy(),
    #         },
    #         metrics={
    #             "Difficulty": metrics.SparseCategoricalAccuracy(),
    #         },
    #     )
    #
    #     difficulty_data = random.randint(0, 3, size=(num_samples, 1))  # 0, 1, 2
    #     new_difficulty_model.fit(
    #         {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
    #         {"Difficulty": difficulty_data},
    #         epochs=1,
    #     )
    #     new_difficulty_model.evaluate(
    #         {"Title": title_data, "Text body": text_body_data, "Tags": tags_data},
    #         {"Difficulty": difficulty_data},
    #     )
    #     utils.plot_model(
    #         new_difficulty_model,
    #         "topology-new-difficulty-model.png",
    #         show_layer_names=True,
    #         show_shapes=True,
    #     )

    # incremental_model(model)


def main():
    simple()
    multiple_input_and_output()

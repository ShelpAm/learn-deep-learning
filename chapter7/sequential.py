import keras
from chapter7.config import dense_names, input_shape, model_name


def main():
    # Two identical ways of defining sequential model
    model = keras.Sequential(
        [
            # Comment off the following line to eagerly declare input shape.
            # A specified, but implicit layer.
            # keras.Input(shape=(input_shape,)),
            keras.layers.Dense(
                64, activation=keras.activations.relu, name=dense_names[0]
            ),
            keras.layers.Dense(
                10, activation=keras.activations.softmax, name=dense_names[1]
            ),
        ],
        name=model_name,
    )

    model = keras.Sequential(name=model_name)
    # The same with `keras.Input`; they are the same symbol.
    model.add(keras.layers.Input(shape=(input_shape,)))
    model.add(
        keras.layers.Dense(64, activation=keras.activations.relu, name=dense_names[0])
    )
    model.add(
        keras.layers.Dense(
            10, activation=keras.activations.softmax, name=dense_names[1]
        )
    )

    # assert len(model.weights) == 0

    # `None` here tells the model that the batch size can be any thing.
    # And `None` can't be pass into `keras.Input` because that doesn't accept a
    # `batch size` parameters, and that is implicitly defined to be "accepting
    # every data with any batch size".
    model.build(input_shape=(None, input_shape))

    print("\nWeights of the built model:")
    for weight in model.weights:
        print(weight)

    print()
    model.summary()

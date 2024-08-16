from keras import Sequential, activations, layers

from .config import dense_names, input_shape, model_name


def main():
    # Two identical ways of defining sequential model
    model = Sequential(
        [
            # Comment off the following line to eagerly declare input shape.
            # A specified, but implicit layer.
            # keras.Input(shape=(input_shape,)),
            layers.Dense(64, activation=activations.relu, name=dense_names[0]),
            layers.Dense(10, activation=activations.softmax, name=dense_names[1]),
        ],
        name=model_name,
    )

    model = Sequential(name=model_name)
    # The same with `keras.Input`; they are the same symbol.
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(64, activation=activations.relu, name=dense_names[0]))
    model.add(layers.Dense(10, activation=activations.softmax, name=dense_names[1]))

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

import numpy as np

# Parameters of my model

n = 100
m = 4
num_iterations = int(1e4)
alpha = 1e-1


def accuracy(P, Q):
    return 100 - np.mean(np.abs(Q - P)) * 100


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def randomized_samples(n, m, parameters):
    x = np.random.uniform(-100, 100, size=(n, m))
    x = x / np.linalg.norm(x, axis=0, keepdims=True)  # Normalizes x
    y = full_predict(parameters, x)

    return x, y


def randomized_parameters(bound, next_num_features: int, num_features: int):
    w = np.random.uniform(-bound, bound, size=(next_num_features, num_features))
    b = np.zeros((next_num_features, 1))
    return w, b


def full_predict(parameters, x):
    result = forward_propagate(parameters, x)[-1]
    return result


def predict(w, b, x, activation: str):
    z = w @ x + b
    if activation == "sigmoid":
        a = sigmoid(z)
    elif activation == "tanh":
        a = np.tanh(z)
    else:
        raise Exception("No such activation function")
    return a


def forward_propagate(parameters, x):
    A = [x]
    for w, b in zip(parameters[0], parameters[1]):
        a = A[-1]
        # Shape of a: (next_num_features, num_examples)
        next_a = predict(w, b, a, "sigmoid")
        A.append(next_a)
    return A


def propagate(x, y, a):
    m = x.shape[1]

    dz = a - y
    dw = 1 / m * (dz @ x.T)
    db = 1 / m * dz.sum()
    gradients = (dw, db)
    cost = -1 / m * (y * np.log(a) + (1 - y) * np.log(1 - a)).sum()
    return gradients, cost


def back_propagate(parameters, A, y):
    num_layers = len(parameters)
    costs = []
    w = []
    b = []
    for j in range(num_layers, 0, -1):
        (dw, db), cost = propagate(A[j - 1], y, A[j])

        costs.append(cost)
        w.append(dw)
        b.append(db)

    w = np.array(w).reshape(num_layers, -1, -1)
    b = np.array(b).reshape(num_layers, -1, -1)
    gradients = (list(reversed(w)), list(reversed(b)))
    costs = list(reversed(costs))

    return (gradients, costs)


def optimize(parameters, x, y, num_iterations, learning_rate):
    print()
    for i in range(num_iterations):
        a = forward_propagate(parameters, x)
        gradients, costs = back_propagate(parameters, a, y)
        for (w, b), (dw, db) in zip(parameters, gradients):
            w -= learning_rate * dw
            b -= learning_rate * db

        period = num_iterations // 5
        if i % period == 0:
            print(f"Cost after iteration {i}: {costs[-1]}")
            # print(f"dw: {dw.T}, db: {db}")

    return parameters


def run_model(
    training_x,
    training_y,
    test_X,
    test_Y,
    nums_features: list,
    num_iterations,
    learning_rate,
):
    num_layers = len(nums_features)
    nums_features.insert(0, training_x.shape[0])

    untrained_parameters = zip(
        *[
            randomized_parameters(0.01, nums_features[i], nums_features[i - 1])
            for i in range(1, num_layers + 1)
        ]
    )
    print(f"initial parameters: {untrained_parameters}")
    trained_parameters = optimize(
        untrained_parameters, training_x, training_y, num_iterations, learning_rate
    )

    predicted_training_Y = full_predict(trained_parameters, training_x)
    predicted_test_Y = full_predict(trained_parameters, test_X)

    print(f"\nTraining accuracy: {accuracy(predicted_training_Y, training_y)}%")
    print(f"Test accuracy: {accuracy(predicted_test_Y, test_Y)}%")

    return trained_parameters


def run():
    initialize_default = (
        True
        if str(input("Would you like to use default parameters? [Y/n] ")).lower() != "n"
        else False
    )

    global n, m, num_iterations, alpha
    if not initialize_default:
        n = int(input("Input the dimension of the x: "))
        m = int(input("Input the number of examples: "))
        num_iterations = int(input("Input the number of learning rounds: "))
        alpha = float(input("Input the learning rate: "))

    w, b = randomized_parameters(1, 1, n)
    original_parameters = (w, b)
    # original_parameters = [randomized_parameters(2, n), randomized_parameters(1, 2)]

    x1, y1 = randomized_samples(n, m, original_parameters)
    x2, y2 = randomized_samples(n, m, original_parameters)

    print("\nGenerated x1 and y1:")
    print(x1)
    print(y1)

    trained_parameters = run_model(
        training_x=x1,
        training_y=y1,
        test_X=x2,
        test_Y=y2,
        nums_features=[1],
        # nums_features=[2, 1],
        num_iterations=num_iterations,
        learning_rate=alpha,
    )

    print("\nOriginal parameters:")
    print(original_parameters)

    print("\nTrained parameters:")
    print(trained_parameters)

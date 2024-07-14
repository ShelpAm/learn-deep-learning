import numpy as np

# Parameters of my model

n = 3
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


def randomized_parameters(next_num_features: int, num_features: int):
    w = np.random.uniform(size=(next_num_features, num_features)).T
    b = np.random.uniform(size=(next_num_features, 1))
    return w, b


def full_predict(parameters, x):
    result = forward_propagate(parameters, x)[-1]
    return result


def predict(w, b, x):
    result = sigmoid(w.T @ x + b)
    return result


def propagate(x, y, a):
    m = x.shape[1]

    cost = -1 / m * (y * np.log(a) + (1 - y) * np.log(1 - a)).sum()
    dw = 1 / m * (x @ (a - y).T)
    db = 1 / m * (a - y).sum()
    gradients = (dw, db)
    return gradients, cost


def forward_propagate(parameters, x):
    num_layers = len(parameters)
    data = [x]
    for j in range(num_layers):
        w, b = parameters[j]
        x = data[j]

        # Shape of a: (next_num_features, num_examples)
        a = predict(w, b, x)
        data.append(a)
    return data


def back_propagate(parameters, data, y, learning_rate):
    num_layers = len(parameters)
    costs = []
    for j in range(num_layers - 1, -1, -1):
        w, b = parameters[j]
        x = data[j]
        a = data[j + 1]

        (dw, db), cost = propagate(x, y, a)

        w -= learning_rate * dw
        b -= learning_rate * db
        costs.append(cost)
    return (parameters, list(reversed(costs)))


def optimize(parameters, x, y, num_iterations, learning_rate):
    print()
    for i in range(num_iterations):
        data = forward_propagate(parameters, x)
        parameters, costs = back_propagate(parameters, data, y, learning_rate)

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
    initial_parameters = [
        randomized_parameters(nums_features[i], nums_features[i - 1])
        for i in range(1, num_layers + 1)
    ]
    trained_parameters = optimize(
        initial_parameters, training_x, training_y, num_iterations, learning_rate
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

    original_parameters = [randomized_parameters(2, n), randomized_parameters(1, 2)]

    x1, y1 = randomized_samples(n, m, original_parameters)
    x2, y2 = randomized_samples(n, m, original_parameters)

    print("\nOriginal parameters:")
    print(original_parameters)

    print("\nGenerated x1 and y1:")
    print(x1)
    print(y1)

    trained_parameters = run_model(
        training_x=x1,
        training_y=y1,
        test_X=x2,
        test_Y=y2,
        nums_features=[2, 1],
        num_iterations=num_iterations,
        learning_rate=alpha,
    )

    print("\nTrained parameters:")
    print(trained_parameters)

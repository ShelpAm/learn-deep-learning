import numpy as np

# Parameters of my model

n = 3
m = 1000
t = 10000
alpha = 1e-4


def accuracy(P, Q):
    return 100 - np.mean(np.abs(Q - P)) * 100


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim: int):
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(w.T @ X + b)
    cost = -1 / m * (Y * np.log(A) + (1 - Y) * np.log(1 - A)).sum()
    dw = 1 / m * (X @ (A - Y).T)
    db = 1 / m * (A - Y).sum()
    grads = (dw, db)
    return grads, cost


def optimize(w, b, X, Y, num_iteration, learning_rate):
    for i in range(num_iteration):
        grads, cost = propagate(w, b, X, Y)
        dw, db = grads

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 500 == 0:
            print(f"Cost after iteration {i}: {cost}")
    return w, b


def predict(w, b, X):
    return np.round(sigmoid(w.T @ X + b))


def model(training_X, training_Y, test_X, test_Y, num_iterations, learning_rate):
    w, b = initialize_with_zeros(training_X.shape[0])
    w, b = optimize(w, b, training_X, training_Y, num_iterations, learning_rate)

    # print("Training X:\n", training_X)
    # print("Training Y:\n", training_Y)

    predicted_training_Y = predict(w, b, training_X)
    predicted_test_Y = predict(w, b, test_X)

    print(f"Training accuracy: {accuracy(predicted_training_Y, training_Y)}%")
    print(f"Test accuracy: {accuracy(predicted_test_Y, test_Y)}%")

    return w, b


def run():
    initialize_default = (
        True
        if str(input("Would you like to use default parameters? [Y/n] ")).lower() != "n"
        else False
    )

    global n, m, t, alpha
    if not initialize_default:
        n = int(input("Input the dimension of the x: "))
        m = int(input("Input the number of examples: "))
        t = int(input("Input the number of learning rounds: "))
        alpha = float(input("Input the learning rate: "))

    w = np.random.uniform(-1, 1, size=(n, 1))
    b = np.random.uniform(-1, 1)

    def gen_random_input(n, m):
        x = np.random.uniform(low=-1, high=1, size=(n, m))
        # x = x / np.linalg.norm(x, axis=0, keepdims=True)  # Normalizes x
        y = sigmoid(w.T @ x + b)
        y = np.round(y)

        return x, y

    x1, y1 = gen_random_input(n, m)
    x2, y2 = gen_random_input(n, m)

    print("Generated x1 and y1:\n")
    print(x1)
    print(y1)

    w, b = model(
        training_X=x1,
        training_Y=y1,
        test_X=x2,
        test_Y=y2,
        num_iterations=t,
        learning_rate=alpha,
    )

    print("Trained parameters:\n")
    print(f"w\n{w}")
    print(f"b\n{b}")

    print()
    print("Original w\n", w)
    print("Original b\n", b)

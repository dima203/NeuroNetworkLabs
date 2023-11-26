from matplotlib import pyplot

from math import sin, cos
import random
import time

from neuro_lib import LinearFunction, SigmoidFunction, NeuroLayer, NeuroNetwork


def predict_func(x: float) -> float:
    return 0.4 * cos(0.4 * x) + 0.08 * sin(0.4 * x)


if __name__ == '__main__':
    random.seed(1)
    input_count = 6
    X = [x / 10 for x in range(30)]
    predict_X = [x / 10 for x in range(len(X), len(X) + 3 * input_count)]
    train_set_x = [list(map(predict_func, X[i:i+input_count])) for i in range(len(X) - input_count)]
    train_set_y = [[predict_func(X[i])] for i in range(input_count, len(X))]
    predicted_x = [y[0] for y in train_set_y]
    predict_set_x = [list(map(predict_func, predict_X[i:i+input_count])) for i in range(len(predict_X) - input_count)]
    predict_set_y = [[predict_func(predict_X[i])] for i in range(input_count, len(predict_X))]

    hidden_layer = NeuroLayer(input_count, 10, SigmoidFunction())
    layer = NeuroLayer(10, 1, LinearFunction())
    neuro_network = NeuroNetwork(input_count, hidden_layer, layer, learning_rate=0.2)
    predicted_y = neuro_network.predict(train_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(X, list(map(predict_func, X)), X[6:], predicted_y)
    pyplot.show()
    errors = neuro_network.learn(train_set_x, train_set_y, error=0.0001)
    pyplot.plot(errors)
    pyplot.show()
    predicted_y = neuro_network.predict(train_set_x)
    print(train_set_y)
    print(predicted_y)
    print([(y[0] - t[0]) ** 2 / 2 for y, t in zip(train_set_y, predicted_y)])
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot([*X, *predict_X], list(map(predict_func, [*X, *predict_X])))
    pyplot.plot(X[6:], predicted_y)
    predicted_y = neuro_network.predict(predict_set_x)
    print(predict_set_y)
    print(predicted_y)
    print([(y[0] - t[0]) ** 2 / 2 for y, t in zip(predict_set_y, predicted_y)])
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(predict_X[6:], predicted_y)
    pyplot.show()

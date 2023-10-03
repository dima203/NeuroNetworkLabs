from matplotlib import pyplot

from math import sin
import random
import time

from neuro_lib import LinearFunction, NeuroLayer, NeuroNetwork


def predict_func(x: float) -> float:
    return sin(9 * x) + 0.5


if __name__ == '__main__':
    random.seed = time.time()
    X = [x / 10 for x in range(31)]
    predict_X = [x / 10 for x in range(26, 45)]
    train_set_x = [list(map(predict_func, X[i:i+4])) for i in range(27)]
    train_set_y = [[predict_func(X[i])] for i in range(4, 31)]
    predicted_x = [y[0] for y in train_set_y]
    predict_set_x = [list(map(predict_func, predict_X[i:i+4])) for i in range(15)]
    predict_set_y = [[predict_func(predict_X[i])] for i in range(4, 19)]

    layer = NeuroLayer(4, 1, LinearFunction())
    neuro_network = NeuroNetwork(4, 0.3, layer)
    predicted_y = neuro_network.predict(train_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(X, list(map(predict_func, X)), X[4:], predicted_y)
    pyplot.show()
    neuro_network.learn(train_set_x, train_set_y, error=0.00001)
    pyplot.show()
    predicted_y = neuro_network.predict(train_set_x)
    print(train_set_y)
    print(predicted_y)
    print([(y[0] - t[0]) ** 2 / 2 for y, t in zip(train_set_y, predicted_y)])
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot([*X, *predict_X[5:]], list(map(predict_func, [*X, *predict_X[5:]])))
    pyplot.plot(X[4:], predicted_y)
    predicted_y = neuro_network.predict(predict_set_x)
    print(predict_set_y)
    print(predicted_y)
    print([(y[0] - t[0]) ** 2 / 2 for y, t in zip(predict_set_y, predicted_y)])
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(predict_X[4:], predicted_y)
    pyplot.show()

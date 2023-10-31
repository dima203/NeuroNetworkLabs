from matplotlib import pyplot

from math import sin, cos
import random
import time

from neuro_lib import LinearFunction, SigmoidFunction, NeuroLayer, NeuroNetwork


def predict_func(x: float) -> float:
    return 0.1 * cos(0.5 * x) + 0.09 * sin(0.5 * x)


if __name__ == '__main__':
    random.seed = time.time()
    input_count = 8
    X = [x for x in range(48)]
    predict_X = [x for x in range(len(X), len(X) + 3 * input_count)]
    train_set_x = [list(map(predict_func, X[i:i+input_count])) for i in range(len(X) - input_count)]
    train_set_y = [[predict_func(X[i])] for i in range(input_count, len(X))]
    predicted_x = [y[0] for y in train_set_y]
    predict_set_x = [list(map(predict_func, predict_X[i:i+input_count])) for i in range(len(predict_X) - input_count)]
    predict_set_y = [[predict_func(predict_X[i])] for i in range(input_count, len(predict_X))]

    hiden_layer = NeuroLayer(input_count, 3, SigmoidFunction())
    layer = NeuroLayer(3, 1, LinearFunction())
    neuro_network = NeuroNetwork(input_count, hiden_layer, layer)
    predicted_y = neuro_network.predict(train_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(X, list(map(predict_func, X)), X[8:], predicted_y)
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
    pyplot.plot(X[8:], predicted_y)
    predicted_y = neuro_network.predict(predict_set_x)
    print(predict_set_y)
    print(predicted_y)
    print([(y[0] - t[0]) ** 2 / 2 for y, t in zip(predict_set_y, predicted_y)])
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(predict_X[8:], predicted_y)
    pyplot.show()

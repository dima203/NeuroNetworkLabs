from matplotlib import pyplot

from math import sin

from neuro_lib import LinearFunction, NeuroLayer, NeuroNetwork


def predict_func(x: float) -> float:
    return sin(9 * x) + 0.5


if __name__ == '__main__':
    X = [x / 10 for x in range(30)]
    predict_X = [x / 10 for x in range(30, 45)]
    train_set_x = [list(map(predict_func, X[i:i+4])) for i in range(26)]
    train_set_y = [[predict_func(X[i])] for i in range(4, 30)]
    predicted_x = [y[0] for y in train_set_y]
    predict_set_x = [list(map(predict_func, predict_X[i:i+4])) for i in range(11)]
    predict_set_y = [[predict_func(predict_X[i])] for i in range(4, 15)]

    layer = NeuroLayer(4, 1, LinearFunction())
    neuro_network = NeuroNetwork(4, 0.2, layer)
    predicted_y = neuro_network.predict(train_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(X, list(map(predict_func, X)), X[4:], predicted_y)
    pyplot.show()
    neuro_network.learn(train_set_x, train_set_y, 5)
    predicted_y = neuro_network.predict(train_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot([*X, *predict_X], list(map(predict_func, [*X, *predict_X])))
    pyplot.plot(X[4:], predicted_y)
    predicted_y = neuro_network.predict(predict_set_x)
    predicted_y = [y[0] for y in predicted_y]
    pyplot.plot(predict_X[4:], predicted_y)
    pyplot.show()

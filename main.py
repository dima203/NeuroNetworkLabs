from neuro_lib import BipolarStepFunction, NeuroLayer, NeuroNetwork


if __name__ == '__main__':
    train_set_x = [
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ]

    train_set_y = [
        [1],
        [1],
        [-1],
        [-1],
    ]

    layer = NeuroLayer(2, 1, BipolarStepFunction())
    neuro_network = NeuroNetwork(2, 0.7, layer)
    print(neuro_network.predict(train_set_x))
    print(layer.get_weights())
    for _ in range(5):
        neuro_network.learn(train_set_x, train_set_y, 1)
        print(neuro_network.predict(train_set_x))
        print(layer.get_weights())

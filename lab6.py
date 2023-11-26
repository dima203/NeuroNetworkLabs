from neuro_lib import HopfieldNetwork, HopfieldLayer, StepFunction, BipolarStepFunction
from copy import deepcopy
from random import randint


vector1 = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]
vector2 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
vector3 = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
vector4 = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
vector5 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
vector6 = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
vector7 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
vector8 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]

inputs = [vector5, vector4, vector8]

layer = HopfieldLayer(20, StepFunction())
network = HopfieldNetwork(layer)
network.learn(inputs)

inputs_vectors = deepcopy(inputs)
result = network.predict(inputs_vectors)
inverse_count = 0
while all(i == j for i, j in zip(inputs, result)):
    print(inverse_count)
    for vector in inputs_vectors:
        rand_position = randint(0, len(vector) - 1)
        vector[rand_position] = 0 if vector[rand_position] == 1 else 1
    result = network.predict(inputs_vectors)
    inverse_count += 1
else:
    for i, in_vector, j in zip(inputs, inputs_vectors, result):
        i_str, in_str, j_str = '', '', ''
        for i_value, in_value, j_value in zip(i, in_vector, j):
            if i_value != j_value:
                i_str += f' {i_value} '
                in_str += f' {in_value} '
                j_str += f'*{j_value}*'
            else:
                i_str += ' ' + str(i_value) + ' '
                in_str += f' {in_value} '
                j_str += ' ' + str(j_value) + ' '
        print(i_str, '\t', in_str, ' -> ', j_str)

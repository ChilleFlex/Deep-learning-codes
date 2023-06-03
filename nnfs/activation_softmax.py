import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() # =np.random.seed(0)

inputs, y = spiral_data(samples= 100,classes= 3)

class dense_layer:
    def __init__(self, n_input, n_neuron):
        self.n_weight = 0.1* np.random.randn(n_input, n_neuron)
        self.n_bias = np.zeros((1, n_neuron))
        
    def forward(self, input):
        self.output = np.dot(input, self.n_weight)+ self.n_bias

class Activation_Relu:
    def actv_func(self, input):
        self.output = np.maximum(0, input)
        
class Activation_Softmax:
    def actv_func(self, input):
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values/ np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
# the spiral_data has 2 attributes     
layer1 = dense_layer(2, 3)
layer1.forward(inputs)
# print(f"Output before activation is applied:\n {layer1.output}")

relu = Activation_Relu()
relu.actv_func(layer1.output)
# print(f"Output after activation is applied:\n {relu.output}")

layer2 = dense_layer(3, 3)  
layer2.forward(relu.output)
softmax = Activation_Softmax()
softmax.actv_func(layer2.output)

print(f"Output after softmax activation is applied:\n {softmax.output[:3]}")
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import math

nnfs.init() # =np.random.seed(0)

e = math.e
inputs, y = spiral_data(samples= 100,classes= 3)

class dense_layer:
    def __init__(self, n_input, n_neuron):
        self.n_weight = 0.01* np.random.randn(n_input, n_neuron)
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
        
class Loss: 
    def calculate(self, output, target):
        sample_losses = self.forward(output, target)
        return np.mean(sample_losses)
    
class Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        input_batch_len = len(y_pred)
        clipped_y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
        correct_confidences = clipped_y_pred[range(input_batch_len), y_true]
        loss = -np.log(correct_confidences)
        return loss
    
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

loss_fn = Categorical_Cross_Entropy()
loss = loss_fn.calculate(softmax.output, y)

print(f"Loss: {loss}")
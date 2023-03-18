import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
]

class dense_layer:
    def __init__(self, n_input, n_neuron):
        self.n_weight = 0.1* np.random.randn(n_input, n_neuron)
        self.n_bias = np.zeros((1, n_neuron))
        
    def forward(self, input):
        self.output = np.dot(input, self.n_weight)+ self.n_bias

# structure -> 4*5*2      
layer1 = dense_layer(4, 5)
layer1.forward(inputs)
print(f"layer1 output: {layer1.output}")

layer2 = dense_layer(5, 2)
layer2.forward(layer1.output) # input for this layer is the output of the first layer
print(f"layer2 output: {layer2.output}")
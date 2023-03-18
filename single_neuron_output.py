inputs = [1.2, 5.1, 2.0]
weights = [3.4, 2.6, 1.1]
bias = 3

# 3*1 -> output from a neuron with 3 inputs
output = inputs[0]*weights[0] + inputs[1]*weights[1]+ inputs[2]*weights[2] + bias
print(output)
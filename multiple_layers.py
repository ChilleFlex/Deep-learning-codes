import numpy as np

# Batches of inputs for 4*3(1 HL consists of 3 neurons)
inputs = np.array([[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
])
# 4 inputs so 4 wights will be genrated to the hidden layer 
# 2 neurons in the HL thats why 3 rows
weights1 = np.array([[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
])
bias1 = np.array([2, 3, 0.5])

# another HL is getting added consisting of 3
weights2 = np.array([
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
])
bias2 = np.array([-1, 2, -0.5])

output_layer1 = np.dot(inputs, np.array(weights1).T) + bias1
output_layer2 = np.dot(output_layer1, np.array(weights2).T) + bias2

print(f"output layer 1 = {output_layer1}\noutput layer2 = {output_layer2}")
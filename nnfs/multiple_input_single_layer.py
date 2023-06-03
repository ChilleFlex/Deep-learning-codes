import numpy as np

# Batches of inputs for 4*3(1 HL consists of 3 neurons)
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
]
# 4 inputs so 4 wights will be genrated to the hidden layer 
# 2 neurons in the HL thats why 3 rows
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
bias = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + bias
print(outputs)
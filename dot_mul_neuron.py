import numpy as np

# 4*3 structure
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
bias = [2, 3, 0.5]

output = np.dot(weights, inputs)+bias
print(f"layer outputs: {output}")
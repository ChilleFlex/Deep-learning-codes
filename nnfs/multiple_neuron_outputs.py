# 4*3 structure
inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
bias = [2, 3, 0.5]

layer_output = []
for n_bias, n_weights in zip(bias, weights):
    n_output = 0
    for weight, n_inputs in zip(n_weights, inputs):
        n_output += weight*n_inputs
    n_output += n_bias
    layer_output.append(n_output)
    
print(f"layer outputs: {layer_output}")
    
        
    


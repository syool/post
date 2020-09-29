import numpy as np

# AND Operation
and_op = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1]])
and_x = and_op[0:4, :-1] # input data
and_Y = and_op[0:4, 2:] # labeled data

# OR Operation
or_op = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
or_x = or_op[0:4, :-1] # input data
or_Y = or_op[0:4, 2:] # labeled data

# parameters
w = np.random.randn(1, 2)
b = 1

learning_rate = 0.1

# activation function
def step_function(x):
    x = 0 if x <= 0 else 1
    return x

# perceptron
epoch = 10
tmp, y = [], []
for repeat in range(epoch):
    for i in range(len(and_op)):
        p1 = and_x[i][0] * w[0][0]
        p2 = and_x[i][1] * w[0][1]
        tmp.append(p1 + p2 + b)
        y.append(step_function(tmp[i]))
        
        print(step_function(tmp[i]))

print(and_Y)

print(w[0][0])
print(y[0])
w[0][0] = w[0][0] + learning_rate * (y[0] - and_Y[0]) * and_x[0][0]
print(w[0][0])
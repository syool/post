import numpy as np

# AND Operation
and_op = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 1, 1]])
and_x = and_op[0:4, :-1] # input data
and_Y = and_op[0:4, 2:].flatten() # labeled data

# OR Operation
or_op = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])
or_x = or_op[0:4, :-1] # input data
or_Y = or_op[0:4, 2:].flatten() # labeled data

# parameters
w = np.random.randn(1, 2)

print("w:", w)

learning_rate = 0.01

# activation function
def step_function(x):
    x = 0 if x <= 0 else 1
    return x

# perceptron
tmp, y = [], []
for i in range(len(and_Y)):
    p1 = and_x[i][0] * w[0][0]
    p2 = and_x[i][1] * w[0][1]
    tmp.append(p1 + p2)
    y.append(step_function(tmp[i]))

print(y)
print(and_Y)
print(and_x)

print('before training w1: ', w[0][0], ' w2: ', w[0][1])

for epoch in range(0, 10):
    for i in range(len(w)):
        for j in range(len(and_Y)):
            w[0][i] = w[0][i] + learning_rate * ((y[j] - and_Y[j])**2)
            print('w1: ', w[0][0], ' w2: ', w[0][1])

answer = []
for i in range(len(and_Y)):
    p1 = and_x[i][0] * w[0][0]
    p2 = and_x[i][1] * w[0][1]
    tmp.append(p1 + p2)
    answer.append(step_function(tmp[i]))

print(answer)
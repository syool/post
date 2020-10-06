import numpy as np

class Perceptron(object):

    def __init__(self, no_of_inputs, iteration=10, learning_rate=0.01):
        self.iter = iteration
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1) # weights + bias
           
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0] # weights[0] are biases
        if summation > 0:
          activation = 1
        else:
          activation = 0            
        return activation

    def train(self, training_inputs, labels):
        for i in range(self.iter):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            print('iteration {}: {}'.format(i+1, self.weights))


training_inputs = []
for i in range(2):
    for j in range(2):
        training_inputs.append(np.array([i, j]))


# ===== training 1: AND operation =====
print('Training data: ', training_inputs)
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

# prediction
for i in range(2):
    for j in range(2):
        inputs = np.array([i, j])
        print(perceptron.predict(inputs))


# ===== training 2: OR operation =====
print('Training data: ', training_inputs)
labels = np.array([0, 1, 1, 1])

perceptron = Perceptron(2)
perceptron.train(training_inputs, labels)

# prediction
for i in range(2):
    for j in range(2):
        inputs = np.array([i, j])
        print(perceptron.predict(inputs))
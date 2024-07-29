from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class KNN:
  def __init__(self, k=5):
    self.k = k

  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def knn_algorithm(self,current_point):
    '''
    This function performs the KNN algorithm.
    '''
    distances = []
    for i in range(self.X_train.shape[0]):
      distance = np.sqrt(np.sum((self.X_train[i] - current_point)**2))
      distances.append((distance, self.y_train[i]))

    # Sort the distance list in ascending order and select first k distances
    nearest_neightbor = sorted(distances)[:self.k]

    # Get the most common class among the neighbors
    classes = []
    class_count = {}
    for neighbor in nearest_neightbor:
      # Class label is 2nd element
      classes.append(neighbor[1])

    for class_label in classes:
      if class_label in class_count:
        class_count[class_label] += 1
      else:
        class_count[class_label] = 1

    # The label with the most occurences is teh prediction
    prediction = max(class_count, key=class_count.get)

    return prediction

  def predict(self, X_test):
    '''
    Predict the class labels for the test data
    '''
    predictions = []

    for point in X_test:
      prediction = self.knn_algorithm(point)
      predictions.append(prediction)

    return predictions

  def accuracy(self, X_test, y_test):
    '''
    Calculate the accuracy
    '''
    y_pred = self.predict(X_test)
    return np.mean(y_pred == y_test) * 100

class LogisticRegression:
  def __init__(self, learning_rate=0.01, n_epochs=1000):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs

  def sigmoid(self, z):
    '''
    Activation function that takes input and outputs in the value of 0 to 1
    '''
    return 1 / (1 + np.exp(-z))

  def loss(self, y_true, y_pred):
    '''
    Loss function to measure how well the weights are performing
    '''
    y_pred = np.clip(y_pred, -500, 500)
    return np.mean(-y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

  def fit(self, X, y):
    '''
    Fit the logistic regression model to the training data
    '''
    self.weights = np.random.rand(X.shape[1])

    for epoch in range(self.n_epochs):
      y_pred = self.sigmoid(np.dot(X, self.weights))

      # Compute gradients
      gradient = np.dot(X.T, (y_pred - y)) / y.size

      # Update weights
      self.weights -= self.learning_rate * gradient

  def predict(self, X):
    '''
    Predict the class labels for the test data
    '''
    y_pred = self.sigmoid(np.dot(X, self.weights))
    return (y_pred >= 0.5).astype(int)

  def accuracy(self, X, y):
    '''
    Calculate the accuracy
    '''
    y_pred = self.predict(X)
    return np.mean(y_pred == y) * 100

class NeuralNetwork:
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, n_epochs=1000):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs

    # Initialize weights and biases
    self.w1 = np.random.rand(self.input_size, self.hidden_size)
    self.w2 = np.random.rand(self.hidden_size, self.output_size)
    self.b1 = np.zeros((1, self.hidden_size))
    self.b2 = np.zeros((1, self.output_size))

  def sigmoid(self, x):
    '''
    Activation function that takes input and outputs in the value of 0 to 1
    '''
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, z):
    '''
    Derivative of the sigmoid function
    '''
    return self.sigmoid(z) * (1 - self.sigmoid(z))

  def fit(self, X, y):
    '''
    Fit the neural network to the training data
    '''

    for epoch in range(self.n_epochs):
      # Forward pass
      hidden_input = np.dot(X, self.w1) + self.b1
      hidden_output = self.sigmoid(hidden_input)
      final_input = np.dot(hidden_output, self.w2) + self.b2
      final_output = self.sigmoid(final_input)

      # Backward pass
      output_error = final_output - y.reshape(-1,1)
      output_delta = output_error * self.sigmoid_derivative(final_output)

      hidden_error = np.dot(output_delta, self.w2.T)
      hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

      # Update weights and biases
      self.w1 -= self.learning_rate * np.dot(X.T, hidden_delta) / X.shape[0]
      self.b1 -= self.learning_rate * np.mean(hidden_delta, axis=0)
      self.w2 -= self.learning_rate * np.dot(hidden_output.T, output_delta) / X.shape[0]
      self.b2 -= self.learning_rate * np.mean(output_delta, axis=0)

  def predict(self, X):
    '''
    Predict the class labels for the test data
    '''
    hidden_input = np.dot(X, self.w1) + self.b1
    hidden_output = self.sigmoid(hidden_input)
    final_input = np.dot(hidden_output, self.w2) + self.b2
    final_output = self.sigmoid(final_input)
    return (final_output >= 0.5).astype(int)

  def accuracy(self, X, y):
    '''
    Calculate the accuracy
    '''
    y_pred = self.predict(X)
    return np.mean(y_pred == y) * 100

# KNN Accuracy
knn_model = KNN(k=5)
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.accuracy(X_test, y_test)
print(f"---------- KNN ACCURACY SCORE ----------\n {knn_accuracy: .2f}%")

# Logistic Regression Accuracy
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.accuracy(X_test, y_test)
print(f"---------- LOGISTIC REGRESSION ACCURACY SCORE ----------\n {lr_accuracy: .2f}%")

# Neural Network Accuracy
nn_model = NeuralNetwork(input_size=X_train.shape[1], hidden_size=30, output_size=1)
nn_model.fit(X_train, y_train)
nn_accuracy = nn_model.accuracy(X_test, y_test)
print(f"---------- NEURAL NETWORK ACCURACY SCORE ----------\n {nn_accuracy: .2f}%")

ensemble_accuracy = (knn_accuracy + lr_accuracy + nn_accuracy) / 3
print(f"---------- ENSEMBLE ACCURACY SCORE ----------\n {ensemble_accuracy: .2f}%")
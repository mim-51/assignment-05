#Here’s a complete Python script that implements logistic regression for binary classification, predicting whether a student is admitted based on exam scores. The script follows your specified tasks:
import numpy as np
import matplotlib.pyplot as plt

# Generate random dataset
np.random.seed(0)
m = 100  # Number of training examples
X = np.random.randn(m, 2) * 10 + 50  # Exam scores (normalized)
y = (X[:, 0] + X[:, 1] > 100).astype(int)  # Admission (0 or 1)

# Visualize data using a scatter plot
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
# plt.xlabel("Exam 1 Score")
# plt.ylabel("Exam 2 Score")
# plt.title("Admission based on Exam Scores")
# plt.show()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(theta, X, y):
    m = len(y)
    h = sigmoid(X.dot(theta))
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient descent
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(sigmoid(X.dot(theta)) - y)
        theta -= alpha * gradient
    return theta

# Preparing data for training
X_train = np.c_[np.ones(m), X]  # Add bias term
theta_opt = gradient_descent(X_train, y)

# Plot decision boundary
x_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_boundary = -(theta_opt[0] + theta_opt[1] * x_boundary) / theta_opt[2]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.plot(x_boundary, y_boundary, 'k-', label="Decision Boundary")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()

# Evaluate accuracy on the training set
predictions = sigmoid(X_train.dot(theta_opt)) >= 0.5
accuracy = np.mean(predictions == y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
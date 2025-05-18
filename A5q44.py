import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.optimize import minimize

# Generate synthetic microchip test dataset
np.random.seed(42)
m = 100
X = np.random.randn(m, 2) * 2  # Test scores
y = (X[:, 0]**2 + X[:, 1]**2 > 2.5).astype(int)  # Pass/Fail Labels

# Scatter plot of dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel("Test Score 1")
plt.ylabel("Test Score 2")
plt.title("Microchip Test Results")
plt.show()

# Feature mapping function (polynomial terms up to degree 6)
def feature_mapping(X1, X2, degree=6):
    output = [np.ones(X1.shape[0])]
    for i, j in itertools.combinations_with_replacement(range(1, degree + 1), 2):
        output.append((X1 ** i) * (X2 ** j))
    return np.array(output).T

X_mapped = feature_mapping(X[:, 0], X[:, 1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Regularized cost function
def cost_function(theta, X, y, lambd):
    m = len(y)
    h = sigmoid(X.dot(theta))
    reg_term = (lambd / (2 * m)) * np.sum(theta[1:] ** 2)
    return (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + reg_term

# Regularized gradient function
def gradient(theta, X, y, lambd):
    m = len(y)
    h = sigmoid(X.dot(theta))
    grad = (1/m) * X.T.dot(h - y)
    grad[1:] += (lambd / m) * theta[1:]  # Regularization term
    return grad

# Training model with different regularization values
lambdas = [0, 1, 100]  # No regularization, moderate, high regularization
theta_vals = []

for lambd in lambdas:
    initial_theta = np.zeros(X_mapped.shape[1])
    result = minimize(cost_function, initial_theta, args=(X_mapped, y, lambd), 
                      method='TNC', jac=gradient)
    theta_vals.append(result.x)

# Decision boundary visualization
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = feature_mapping(xx.ravel(), yy.ravel())
plt.figure(figsize=(12, 6))

for i, lambd in enumerate(lambdas):
    Z_pred = sigmoid(Z.dot(theta_vals[i])).reshape(xx.shape)
    plt.subplot(1, 3, i+1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.contour(xx, yy, Z_pred, levels=[0.5], linewidths=2, colors='black')
    plt.title(f"Decision Boundary (λ = {lambd})")
    plt.xlabel("Test Score 1")
    plt.ylabel("Test Score 2")

plt.tight_layout()
plt.show()

# Discussion on regularization effects
for i, lambd in enumerate(lambdas):
    print(f"\nλ = {lambd}:")
    if lambd == 0:
        print("No regularization → Overfitting likely (complex boundary).")
    elif lambd == 1:
        print("Moderate regularization → Balanced decision boundary.")
    elif lambd == 100:
        print("High regularization → Underfitting likely (simpler boundary).")
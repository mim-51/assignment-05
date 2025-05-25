import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# 1. Generate dataset
np.random.seed(0)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = (X * np.sin(X) + np.random.randn(100, 1) * 0.5).ravel()

# 2. Polynomial feature mapping
degree = 5
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

# 3. Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# 4. Ridge regression (regularized linear regression)
def train_and_plot(lambda_val):
    model = Ridge(alpha=lambda_val)
    model.fit(X_train, y_train)

    print(f"\nLambda: {lambda_val}")
    print("Learned parameters:", model.coef_)

    # Fit curve
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_pred = model.predict(X_plot_poly)

    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, label='Training data')
    plt.plot(X_plot, y_pred, color='red', label=f'Degree {degree} fit')
    plt.title(f'Polynomial Regression Fit (λ={lambda_val})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Learning curve plotting
def plot_learning_curve(lambda_val):
    train_errors = []
    val_errors = []
    sizes = range(10, len(X_train), 5)

    for m in sizes:
        model = Ridge(alpha=lambda_val)
        model.fit(X_train[:m], y_train[:m])

        y_train_pred = model.predict(X_train[:m])
        y_val_pred = model.predict(X_val)

        train_errors.append(np.mean((y_train_pred - y_train[:m]) ** 2))
        val_errors.append(np.mean((y_val_pred - y_val) ** 2))

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, train_errors, label='Training error')
    plt.plot(sizes, val_errors, label='Validation error')
    plt.title(f'Learning Curves (λ={lambda_val})')
    plt.xlabel('Training set size')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

# 6. Analyze different regularization values
lambda_values = [0, 0.01, 1, 100]
for l in lambda_values:
    train_and_plot(l)
    plot_learning_curve(l)
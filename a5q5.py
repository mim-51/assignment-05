import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
np.random.seed(0)
X = np.sort(np.random.uniform(0, 10, 100)).reshape(-1, 1)
y = (X * np.sin(X)).ravel() + np.random.normal(0, 1, X.shape[0])

# 2. Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# 3. Learning curve function
def plot_learning_curves(X_train, y_train, X_val, y_val, degree, lambda_):
    train_errors = []
    val_errors = []

    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    for m in range(5, len(X_train)+1, 5):
        model = Ridge(alpha=lambda_, fit_intercept=False)
        model.fit(X_train_poly[:m], y_train[:m])

        y_train_predict = model.predict(X_train_poly[:m])
        y_val_predict = model.predict(X_val_poly)

        train_error = mean_squared_error(y_train[:m], y_train_predict)
        val_error = mean_squared_error(y_val, y_val_predict)

        train_errors.append(train_error)
        val_errors.append(val_error)

    plt.plot(range(5, len(X_train)+1, 5), train_errors, label="Train Error")
    plt.plot(range(5, len(X_train)+1, 5), val_errors, label="Val Error")
    plt.title(f"Learning Curves (λ = {lambda_})")
    plt.xlabel("Training Set Size")
    plt.ylabel("Error (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model

# 4. Polynomial Regression Fit Visualization
def plot_polynomial_fit(model, degree, X, y):
    X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree)
    X_plot_poly = poly.fit_transform(X_plot)
    y_plot = model.predict(X_plot_poly)

    plt.scatter(X, y, label="Training Data")
    plt.plot(X_plot, y_plot, color='r', label=f"Polynomial Degree {degree}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Polynomial Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

# 5. Train and analyze for different λ values
for lambda_ in [0, 0.1, 10]:
    print(f"\n--- Lambda: {lambda_} ---")
    degree = 5
    model = plot_learning_curves(X_train, y_train, X_val, y_val, degree, lambda_)
    plot_polynomial_fit(model, degree, X_train, y_train)
    print(f"Learned θ (first 5 terms): {model.coef_[:5]}")
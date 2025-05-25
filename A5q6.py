import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load digits dataset (images of 0-9)
digits = load_digits()
X = digits.data  # shape (1797, 64) — 8x8 pixel images
y = digits.target  # labels (0 to 9)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. One-vs-All Logistic Regression
num_classes = len(np.unique(y_train))
classifiers = []

# Train one classifier per class
for c in range(num_classes):
    # Binary target: 1 if class == c, else 0
    y_binary = (y_train == c).astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_binary)
    classifiers.append(model)

# 4. Predicting with all classifiers
def predict_ova(X):
    probs = np.array([clf.predict_proba(X)[:, 1] for clf in classifiers])  # shape (n_classes, n_samples)
    return np.argmax(probs, axis=0)

y_pred = predict_ova(X_test)

# 5. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"One-vs-All Logistic Regression Accuracy: {accuracy * 100:.2f}%")

# 6. Show learned coefficients (optional)
for i, clf in enumerate(classifiers):
    print(f"Class {i} coefficients shape: {clf.coef_.shape}")

# 7. Visualization of some digits
fig, axes = plt.subplots(2, 5, figsize=(8, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
    ax.set_title(f"Pred: {y_pred[i]}, True: {y_test[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()

# 8. Optional: Comparison using built-in scikit-learn multiclass mode
clf_sklearn = LogisticRegression(max_iter=1000, multi_class='ovr')
clf_sklearn.fit(X_train, y_train)
sklearn_acc = clf_sklearn.score(X_test, y_test)
print(f"\nScikit-learn LogisticRegression Accuracy: {sklearn_acc * 100:.2f}%")
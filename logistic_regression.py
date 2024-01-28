import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, n_iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(n + 1, 1)

    for _ in range(n_iterations):
        logits = X_b.dot(theta)
        predictions = sigmoid(logits)
        errors = predictions - y
        gradients = X_b.T.dot(errors) / m
        theta = theta - learning_rate * gradients

    return theta

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = (X > 1).astype(int)

theta = logistic_regression(X, y)

plt.scatter(X, y, label='Data points')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.plot(X, sigmoid(np.c_[np.ones((100, 1)), X].dot(theta)), color='red', label='Decision Boundary')
plt.legend()
plt.show()

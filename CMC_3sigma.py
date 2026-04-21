import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Linear functions
# ----------------------------
def f1(x, m1, b1):
    return m1 * x + b1

def f2(x, m2, b2):
    return m2 * x + b2

# ----------------------------
# Sigmoid (logistic)
# ----------------------------
def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def sigmoid_derivative(x, k, x0):
    S = sigmoid(x, k, x0)
    return k * S * (1 - S)

# ----------------------------
# Intersection
# ----------------------------
def intersection(m1, b1, m2, b2):
    if m1 == m2:
        raise ValueError("Lines are parallel.")
    return (b2 - b1) / (m1 - m2)

# ----------------------------
# Convert desired sigma → k
# (logistic ≈ Gaussian width match)
# ----------------------------
def sigma_to_k(sigma):
    return np.pi / (np.sqrt(3) * sigma)

# ----------------------------
# Blended function
# ----------------------------
def blended(x, m1, b1, m2, b2, k, x0):
    S = sigmoid(x, k, x0)
    return (1 - S) * f1(x, m1, b1) + S * f2(x, m2, b2)

# ----------------------------
# Full derivative
# ----------------------------
def blended_derivative(x, m1, b1, m2, b2, k, x0):
    S = sigmoid(x, k, x0)
    dS = sigmoid_derivative(x, k, x0)

    return (
        (1 - S) * m1 +
        S * m2 +
        dS * (f2(x, m2, b2) - f1(x, m1, b1))
    )

# ----------------------------
# Second derivative
# ----------------------------
def blended_second_derivative(x, m1, b1, m2, b2, k, x0):
    S = sigmoid(x, k, x0)
    dS = sigmoid_derivative(x, k, x0)

    # Second derivative of sigmoid
    d2S = k * dS * (1 - 2 * S)

    return (
        2 * dS * (m2 - m1) +
        d2S * (f2(x, m2, b2) - f1(x, m1, b1))
    )

# ----------------------------
# Extract Gaussian-like sigma
# ----------------------------
def compute_sigma(x, m1, b1, m2, b2, k, x0):
    g = sigmoid_derivative(x, k, x0) * (f2(x, m2, b2) - f1(x, m1, b1))

    # Ensure positive and normalize
    g = g - np.min(g)
    g /= np.trapz(g, x)

    mu = np.trapz(x * g, x)
    var = np.trapz((x - mu)**2 * g, x)

    return mu, np.sqrt(var)

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Define your two lines
    m1, b1 = 102.8, 0
    m2, b2 = 28.2, 466

    # Domain
    x = np.linspace(-10, 10, 2000)

    # Intersection = center of transition
    x0 = intersection(m1, b1, m2, b2)

    # 🔑 YOU CONTROL SMOOTHNESS HERE
    desired_sigma = 2.0   # increase this → smoother transition

    k = sigma_to_k(desired_sigma)

    print(f"Intersection (center): {x0}")
    print(f"Chosen sigma: {desired_sigma}")
    print(f"Computed k: {k}")

    # Compute functions
    y_blend = blended(x, m1, b1, m2, b2, k, x0)
    y_deriv = blended_derivative(x, m1, b1, m2, b2, k, x0)
    y_second = -blended_second_derivative(x, m1, b1, m2, b2, k, x0)

    mu, sigma_est = compute_sigma(x, m1, b1, m2, b2, k, x0)

    print(f"Extracted mu: {mu}")
    print(f"Extracted sigma: {sigma_est}")

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(x, f1(x, m1, b1), '--', label='f1')
    plt.plot(x, f2(x, m2, b2), '--', label='f2')
    plt.axvline(x0, linestyle=':', label='Intersection')

    plt.plot(x, y_blend, label='Blended')
    plt.plot(x, y_second, label='Second Derivative')
    plt.plot(x, y_deriv, label='Derivative')

    plt.legend()
    plt.title("Smooth Sigmoid Blend with Controlled Width (sigma)")
    plt.show()
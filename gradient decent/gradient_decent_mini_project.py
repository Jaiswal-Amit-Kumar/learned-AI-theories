import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Features
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Debugging configuration
DEBUG = True
PRINT_INTERVAL = 20

def compute_cost(theta, X, y):
    """Compute Mean Squared Error"""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient_descent(X, y, theta, learning_rate=0.1, n_iters=100, 
                    batch_size=None, debug=False):
    """
    Generalized gradient descent implementation
    batch_size: None=full batch, 1=stochastic, 1<size<len(X)=mini-batch
    """
    cost_history = []
    m = len(y)
    
    for iteration in range(n_iters):
        if batch_size:
            # Randomly select batch
            indices = np.random.randint(0, m, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
        else:
            # Full batch
            X_batch = X
            y_batch = y
            
        # Compute gradient
        gradients = (1/m) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Calculate and store cost
        cost = compute_cost(theta, X, y)
        cost_history.append(cost)
        
        # Debugging output
        if debug and iteration % PRINT_INTERVAL == 0:
            print(f"Iteration {iteration}:")
            print(f"Theta: {theta.flatten()}")
            print(f"Gradients: {gradients.flatten()}")
            print(f"Cost: {cost:.4f}")
            print("----------------------------------")
    
    return theta, cost_history

# Initialize parameters
theta_init = np.random.randn(2, 1)

# Run different variants
theta_batch, cost_batch = gradient_descent(
    X_b, y, theta_init.copy(), learning_rate=0.1, 
    n_iters=500, debug=DEBUG, batch_size=None
)

theta_stochastic, cost_stochastic = gradient_descent(
    X_b, y, theta_init.copy(), learning_rate=0.1,
    n_iters=500, debug=DEBUG, batch_size=1
)

theta_minibatch, cost_minibatch = gradient_descent(
    X_b, y, theta_init.copy(), learning_rate=0.1,
    n_iters=500, debug=DEBUG, batch_size=32
)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(cost_batch, 'r-', linewidth=2, label='Batch GD')
plt.plot(cost_stochastic, 'g-', linewidth=1, label='Stochastic GD')
plt.plot(cost_minibatch, 'b-', linewidth=1, label='Mini-Batch GD')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Variants Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Final parameters
print("\nFinal Parameters:")
print(f"Batch GD: {theta_batch.flatten()}")
print(f"Stochastic GD: {theta_stochastic.flatten()}")
print(f"Mini-Batch GD: {theta_minibatch.flatten()}")

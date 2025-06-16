import numpy as np

# Sample data for binary classification
y_true_binary = np.array([1, 0, 1, 1])            # True labels
y_pred_binary = np.array([0.9, 0.2, 0.8, 0.7])    # Predicted probabilities

# Sample data for multi-class classification
y_true_multiclass = np.array([
    [1, 0, 0],  # Class 0
    [0, 1, 0],  # Class 1
    [0, 0, 1],  # Class 2
])
y_pred_multiclass = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6],
])

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def binary_cross_entropy_debug(y_true, y_pred, eps=1e-15):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("epsilon", eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    print("Clipped y_pred = np.clip(y_pred, eps, 1 - eps):", y_pred)
    term1 = y_true * np.log(y_pred)
    term2 = (1 - y_true) * np.log(1 - y_pred)
    print("term1 = y_true * log(y_pred):", term1)
    print("term2 = (1-y_true) * log(1-y_pred):", term2)
    loss = -np.mean(term1 + term2)
    print("Binary Cross-Entropy Loss = -np.mean(term1 + term2):", loss)
    return loss

def categorical_cross_entropy(y_true, y_pred, eps=1e-15):
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return loss

def categorical_cross_entropy_debug(y_true, y_pred, eps=1e-15):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("epsilon", eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    print("Clipped y_pred = np.clip(y_pred, eps, 1 - eps):\n", y_pred)
    log_preds = np.log(y_pred)
    print("Log of y_pred = np.log(y_pred):\n", log_preds)
    product = y_true * log_preds
    print("product = y_true * log(y_pred):\n", product)
    loss = -np.sum(product) / y_true.shape[0]
    print("Categorical Cross-Entropy Loss = -np.sum(product) / y_true.shape[0]:", loss)
    return loss


def hinge_loss(y_true, y_pred):
    # For hinge loss, y_true should be -1 or 1, y_pred is the model's output (not probability)
    loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
    return loss

def hinge_loss_debug(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    margin = 1 - y_true * y_pred
    print("margin = 1 - y_true * y_pred:", margin)
    rectified = np.maximum(0, margin)
    print("Rectified (max(0, ...) = np.maximum(0, margin)):", rectified)
    loss = np.mean(rectified)
    print("Hinge Loss = np.mean(rectified):", loss)
    return loss


def kl_divergence(P, Q, eps=1e-15):
    # P: true distribution, Q: predicted distribution
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    return np.sum(P * np.log(P / Q)) / P.shape[0]

def kl_divergence_debug(P, Q, eps=1e-15):
    print("y_true:", P)
    print("y_pred:", Q)
    print("epsilon", eps)
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    ratio = P / Q
    print("P = np.clip(P, eps, 1) / Q = np.clip(Q, eps, 1):\n", ratio)
    log_ratio = np.log(ratio)
    print("log(P / Q) = np.log(ratio):\n", log_ratio)
    product = P * log_ratio
    print("product = P * log(P / Q):\n", product)
    loss = np.sum(product) / P.shape[0]
    print("KL Divergence = np.sum(product) / P.shape[0]:", loss)
    return loss


# --- Binary Classification Losses ---
print("Binary Cross-Entropy Loss:", binary_cross_entropy(y_true_binary, y_pred_binary))

# For hinge loss, convert binary labels to -1 and 1, and use logits (for demonstration, use probabilities centered at 0)
y_true_hinge = 2 * y_true_binary - 1     # 0 -> -1, 1 -> 1
y_pred_hinge = 2 * y_pred_binary - 1     # scale probabilities to [-1, 1]
print("Hinge Loss (binary):", hinge_loss(y_true_hinge, y_pred_hinge))

# --- Multi-class Classification Losses ---
print("Categorical Cross-Entropy Loss:", categorical_cross_entropy(y_true_multiclass, y_pred_multiclass))

# KL Divergence between true and predicted distributions (multi-class, averaged over samples)
print("KL Divergence (multi-class):", kl_divergence(y_true_multiclass, y_pred_multiclass))

print("------------------------------------------------------------------------------------")
print("Binary Cross-Entropy Loss Debugging Functions:")
print("------------------------------------------------------------------------------------")
binary_cross_entropy_debug(y_true_binary, y_pred_binary)
print("------------------------------------------------------------------------------------")
print("Categorical Cross-Entropy Loss:")
print("------------------------------------------------------------------------------------")
categorical_cross_entropy_debug(y_true_multiclass, y_pred_multiclass)
print("------------------------------------------------------------------------------------")
print("Hinge Loss (binary):")
print("------------------------------------------------------------------------------------")
hinge_loss_debug(y_true_hinge, y_pred_hinge)
print("------------------------------------------------------------------------------------")
print("KL Divergence (multi-class):") 
print("------------------------------------------------------------------------------------")
kl_divergence_debug(y_true_multiclass, y_pred_multiclass)

# --- For visualization or further experimentation, you can modify the sample data above ---

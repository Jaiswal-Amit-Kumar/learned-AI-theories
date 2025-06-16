import numpy as np
import matplotlib.pyplot as plt

# Sample true values and predictions
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.1, 7.8])

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_debug(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    error = y_true - y_pred
    print("Errors = y_true - y_pred:", error)
    squared_error = error ** 2
    print("Squared Errors = error ** 2:", squared_error)
    mse_value = np.mean(squared_error)
    print("MSE mse_value = np.mean(squared_error):", mse_value)
    return mse_value

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_debug(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    errors= y_true - y_pred
    print("MAE errors:", errors)
    absolute = np.abs(y_true - y_pred)
    print("MAE absolute = np.abs(errors):", absolute)
    mean_value = np.mean(absolute)
    print("MAE mean_value = np.mean(absolute):", mean_value)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * error[is_small_error] ** 2
    linear_loss = delta * (np.abs(error[~is_small_error]) - 0.5 * delta)
    return np.mean(np.concatenate([squared_loss, linear_loss]))

def huber_loss_debug(y_true, y_pred, delta=1.0):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    error = y_true - y_pred
    print("Errors = y_true - y_pred:", error)
    is_small_error = np.abs(error) <= delta
    print("Is small error = np.abs(error) <= delta:", is_small_error)
    squared_loss = 0.5 * error[is_small_error] ** 2
    print("Squared Loss = 0.5 * error[is_small_error] ** 2:", squared_loss)
    linear_loss = delta * (np.abs(error[~is_small_error]) - 0.5 * delta)
    print("Linear Loss = delta * (np.abs(error[~is_small_error]) - 0.5 * delta):", linear_loss)
    combined = np.concatenate([squared_loss, linear_loss])
    print("Combined Loss = np.concatenate([squared_loss, linear_loss]):", combined)
    huber = np.mean(combined)
    print("Huber Loss = np.mean(combined):", huber)



def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))

def log_cosh_loss_debug(y_true, y_pred):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    error = y_true - y_pred
    print("log_cosh_loss error = (y_true, y_pred):", error)
    cosh_error = np.cosh(error)
    print("Cosh Errors = np.cosh(error):", cosh_error)
    log_cosh = np.log(cosh_error)
    print("Log-Cosh = np.log(cosh_error):", log_cosh)
    log_cosh_value = np.mean(log_cosh)
    print("Log-Cosh Loss = np.mean(log_cosh):", log_cosh_value)


def quantile_loss(y_true, y_pred, gamma=0.5):
    error = y_true - y_pred
    return np.mean(np.maximum(gamma * error, (gamma - 1) * error))

def quantile_loss_debug(y_true, y_pred, gamma=0.5):
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    error = y_true - y_pred
    print("error = y_true - y_pred:", error)
    term1 = gamma * error
    term2 = (gamma - 1) * error
    print("Term1 (gamma * diff):", term1)
    print("Term2 ((gamma-1) * diff):", term2)
    max_terms = np.maximum(term1, term2)
    print("Max Terms = np.maximum(term1, term2):", max_terms)
    quantile = np.mean(max_terms)
    print("Quantile Loss = np.mean(max_terms):", quantile)


# Calculate and print all losses
print("Mean Squared Error (MSE):", mse(y_true, y_pred))
print("Mean Absolute Error (MAE):", mae(y_true, y_pred))
print("Huber Loss (delta=1):", huber_loss(y_true, y_pred, delta=1.0))
print("Log-Cosh Loss:", log_cosh_loss(y_true, y_pred))
print("Quantile Loss (gamma=0.5, median):", quantile_loss(y_true, y_pred, gamma=0.5))
print("-----------------------------------------------------------------------------------")
print("Mean Squared Error Debugging Functions(MSE):")
mse_debug(y_true, y_pred)
print("------------------------------------------------------------------------------------")
print("Mean Absolute Error (MAE) Debugging Functions:")
mae_debug(y_true, y_pred)
print("------------------------------------------------------------------------------------")
print("Huber Loss Debugging Functions (delta=1):")
huber_loss_debug(y_true, y_pred, delta=1.0)
print("------------------------------------------------------------------------------------")
print("Log-Cosh Losss Debugging Functions:")
log_cosh_loss_debug(y_true, y_pred)
print("------------------------------------------------------------------------------------")
print("Quantile Loss (gamma=0.5, median) Debugging Functions:")
quantile_loss_debug(y_true, y_pred, gamma=0.5)






# # Visualization for a range of predictions
# pred_range = np.linspace(-2, 10, 100)
# losses = {
#     "MSE": [mse(y_true, np.full_like(y_true, p)) for p in pred_range],
#     "MAE": [mae(y_true, np.full_like(y_true, p)) for p in pred_range],
#     "Huber": [huber_loss(y_true, np.full_like(y_true, p), delta=1.0) for p in pred_range],
#     "Log-Cosh": [log_cosh_loss(y_true, np.full_like(y_true, p)) for p in pred_range],
#     "Quantile (0.5)": [quantile_loss(y_true, np.full_like(y_true, p), gamma=0.5) for p in pred_range],
# }

# plt.figure(figsize=(10, 6))
# for name, loss_values in losses.items():
#     plt.plot(pred_range, loss_values, label=name)
# plt.xlabel("Prediction Value")
# plt.ylabel("Loss")
# plt.title("Regression Loss Functions")
# plt.legend()
# plt.grid(True)
# plt.show()



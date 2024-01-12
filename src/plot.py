import matplotlib.pyplot as plt
import numpy as np


def plot_diff(ytrue, ypred):
    true_values = ytrue
    pred_values = ypred

    plt.figure(figsize = (15, 5))
    plt.grid(True, linestyle = '--')
    plt.scatter(np.arange(len(true_values)), true_values, color='red', label='True Values')
    plt.scatter(np.arange(len(pred_values)), pred_values, color='blue', label='Predicted Values')

    # Connect true and predicted values with dotted lines
    for num, (true, pred) in enumerate(zip(true_values, pred_values)):
        plt.plot([num, num], [true, pred], '--', color='gray', linewidth=1)

    plt.legend()
    plt.xlabel('Data Point')
    plt.ylabel('Values')
    plt.title('True vs Predicted Values with Dotted Lines')
    plt.show()
    

def plot_predictions(ytrue, ypred):
    # Sample true values and predicted values
    true_values = ytrue
    pred_values = ypred

    plt.figure(figsize=(13, 10))
    plt.grid(True, linestyle='--')
    # Create a scatter plot
    plt.scatter(true_values, pred_values, color='blue', label='True vs. Predicted')

    # Connect true and predicted values with dotted lines
    for true, pred in zip(true_values, pred_values):
        plt.plot([true, true], [true, pred], '--', color='gray', linewidth=1)

    # Add diagonal reference line in red
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--', label='Reference Line')

    # Add labels and legend
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

    # Show the plot
    plt.show()
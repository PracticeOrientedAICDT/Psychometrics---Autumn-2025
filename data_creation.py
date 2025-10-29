import numpy as np
import pandas as pd


def irt_prob(theta, a, b, c):
    """Calculate the probability of a correct response using the 3PL model."""
    exp_term = np.exp(-a * (theta - b))
    prob = c + (1 - c) / (1 + exp_term)
    return prob

def generate_irt_data(num_students, num_items, a_params, b_params, c_params):
    """Generate synthetic IRT data for students and items."""
    # Simulate student abilities from a standard normal distribution
    theta = np.random.normal(0, 1, num_students)
    
    # Initialize response matrix
    responses = np.zeros((num_students, num_items))
    
    # Generate responses based on IRT probabilities
    for i in range(num_items):
        for j in range(num_students):
            p_correct = irt_prob(theta[j], a_params[i], b_params[i], c_params[i])
            responses[j, i] = np.random.binomial(1, p_correct)
    
    return pd.DataFrame(responses, columns=[f'Item_{i+1}' for i in range(num_items)])

# Example usage
num_students = 10000
num_items = 8
#a_params = np.array([1.0] * num_items)  # Discrimination parameters
a_params = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])  # Discrimination parameters
#b_params = np.linspace(-2, 2, num_items)  # Difficulty parameters
#b_params = np.array([0.0] * num_items)  # Difficulty parameters
b_params = np.array([-2.0, -2.0, -1.0, -1.0, 1.0, 1.0, 1.5, 1.5])  # Difficulty parameters
c_params = np.array([0.0] * num_items)  # Guessing parameters
data = generate_irt_data(num_students, num_items, a_params, b_params, c_params)
data.to_csv('irt_data.csv', index=False)
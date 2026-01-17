"""
Calculate the different VaR metrics for given data.
"""

import numpy as np
import pandas as pd
from scipy import stats


def historical_var(returns, confidence_level=0.95):
    """
    Calculate Historical VaR
    """

    # Convert to numpy array if input is a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    # Calculate alpha and VaR
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)

    return var


def parametric_var(returns, confidence_level=0.95):
    """
    Calculate Parametric VaR
    """

    # Convert to numpy array if input is a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    # Calculate mean and standard deviation
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Calculate alpha and z-score for the given confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(alpha)

    # Calculate VaR using the parametric formula
    var = mu + z_score * sigma

    return var


def monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000):
    """
    Calculate Monte Carlo VaR
    """

    # Convert to numpy array if input is a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    # Calculate mean and standard deviation
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Simulate future returns using a normal distribution
    simulated_returns = np.random.normal(mu, sigma, num_simulations)

    # Calculate alpha and VaR from the simulated returns
    alpha = 1 - confidence_level
    var = np.percentile(simulated_returns, alpha * 100)

    return var


def conditional_var(returns, var, confidence_level=0.95):
    """
    Calculate Conditional VaR (CVaR)
    """

    # Convert to numpy array if input is a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    # Calculate CVaR as the average of losses exceeding VaR
    tail_losses = returns[returns <= var]
    cvar = np.mean(tail_losses)

    return cvar


def compare_var_methods(returns, confidence_level=0.95):
    """
    Display outcome of all VaR methods
    """

    # calculate the different VaR metrics
    historical = historical_var(returns, confidence_level)
    parametric = parametric_var(returns, confidence_level)
    monte_carlo = monte_carlo_var(returns, confidence_level)

    # return the results as a dictionary
    return {
        'Historical VaR': historical,
        'Parametric VaR': parametric,
        'Monte Carlo VaR': monte_carlo
    }


def calculate_var_dollars(var_percentage, portfolio_value):
    """
    Convert VaR from percentage to dollar amount.
    """

    # Multiply percentage VaR by portfolio value to get dollar VaR
    return var_percentage * portfolio_value
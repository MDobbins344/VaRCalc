# calculate the different VaR metrics for given data

import numpy as np
import pandas as pd
from scipy import stats

'''Historical VaR'''
def historical_var(returns, confidence_level=0.95):

    # Convert to numpy array if input is a pandas Series
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    # Calculate alpha and VaR
    alpha = 1 - confidence_level
    var = np.percentile(returns, alpha * 100)

    return var

'''Parametric VaR'''
def parametric_var(returns, confidence_level=0.95):

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

'''Monte Carlo VaR'''
def monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000):

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

'''Display outcome of all VaR methods'''
def compare_var_methods(returns, confidence_level=0.95):
    historical = historical_var(returns, confidence_level)
    parametric = parametric_var(returns, confidence_level)
    monte_carlo = monte_carlo_var(returns, confidence_level)

    return {
        'Historical VaR': historical,
        'Parametric VaR': parametric,
        'Monte Carlo VaR': monte_carlo
    }
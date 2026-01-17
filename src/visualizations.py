"""
Create various plots to visualize VaR data.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats

# setting plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12,6)
plt.rcParams['font.size'] = 12

def plot_return_distribution_with_var(df, var_threshold):
    """
    Plot the return distribution and mark the VaR threshold.
    
    """
    sns.histplot(df['returns'], kde=True, color='blue', bins=50)
    plt.axvline(x=var_threshold, color='red', linestyle='--', label=f'VaR Threshold: {var_threshold}')
    plt.title('Return Distribution with VaR Threshold')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

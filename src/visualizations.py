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

def plot_distribution_with_var(returns, var_value, confidence_level=0.95,
                               cvar_value=None, method_name="Historical"):
    """
    Plot the return distribution and mark the VaR threshold.
    """

    # convert to numpy arry if needed
    if isinstance(returns, pd.Series):
        returns = returns.values
    else:
        returns_array = returns

    # create figure
    fig, ax = plt.subplots(figsize=(12,7))

    # plot histogram of returns
    n, bins, patches = ax.hist(returns_array, bins=50, alpha=0.7, color='skyblue',
                                edgecolor='black', label='Return Distribution')

    # color the tail area for VaR
    for i, patch in enumerate(patches):
        if bins[i] < var_value:
            patch.set_facecolor('salmon')
            patch.set.set_alpha(0.8)

    # plot VaR line
    ax.axvline(var_value, color='red', linestyle='--', linewidth=2.5,
               label=f'{method_name} VaR ({confidence_level:.0%}): {var_value:.4f}')
    
    # plot CVaR line if provided
    if cvar_value is not None:
        ax.axvline(cvar_value, color='orange', linestyle='--', linewidth=2.5,
                   label=f'CVaR ({confidence_level:.0%}): {cvar_value:.4f}')
        
    # add mean line
    mean_return = np.mean(returns_array)
    ax.axvline(mean_return, color='green', linestyle='-', linewidth=2,
               label=f'Mean Return: {mean_return:.4f}')
    
    # add titles and labels
    ax.set_xlabel('Daily Return', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'{method_name} VaR - Return Distribution\n'
                 f'{confidence_level:.0%} Confidence Level',
                 fontsize=14, fontweight='bold')
    
    # format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

    # add legend
    ax.legend(loc='upper right', fontsize=10)

    # add textbox with summary interpretation
    textstr = f'Interpretation:\n'
    textstr += f'• Red area: Worst {(1-confidence_level):.0%} of days\n'
    textstr += f'• {confidence_level:.0%} of days had returns better than {var_value:.2%}\n'
    if cvar_value is not None:
        textstr += f'• Average loss on worst days: {cvar_value:.2%}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_var_comparison(var_results, confidence_level=0.95):
    """
    Plot a comparison of VaR values from the different VaR methods.
    """

    # create figure
    fig, ax = plt.subplots(figsize=(10,6))

    # extract data from parameters
    methods = list(var_results.keys())
    values = [var_results[m] for m in methods]

    # create color palette using all reds or similar
    colors = ['#d62728', '#ff7f0e', '#9467bd'][:len(methods)]

    # create bar chart
    bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor='black',
                  linewidth=1.5)
    
    # add value labels on top of bars
    for bar, value in zip(bar, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom' if value < 0 else 'top',
                  fontsize=11, fontweight='bold')
        
    # add horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # add titles and labels
    ax.set_ylabel('VaR (Potential Loss)', fontsize=12, fontweight='bold')
    ax.set_title(f'VaR Method Comparison\n'
                 f'{confidence_level:.0%} Confidence Level',
                 fontsize=14, fontweight='bold')
    
    # format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))

    # add grid
    ax.grid(axis='y', alpha=0.3)

    # add textbox with summary interpretation
    avg_var = np.mean(values)
    textstr = f'Average VaR: {avg_var:.2%}\n'
    textstr += f'Range: {min(values):.2%} to {max(values):.2%}\n'

    abs_values = [abs(v) for v in values]
    range_diff = max(abs_values) - min(abs_values)
    if range_diff < 0.01: # less than 1%
        textstr += f'\nAll methods show similar risk levels'
    elif range_diff < 0.02: # less than 2%
        textstr += f'\nMethods show moderately different risk levels'
    else: # more than 2%
        textstr += f'\nMethods show significantly different risk levels'
        textstr += f'\n(Suggests non-normal returns or model risk)'

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    return fig

def plot_portfolio_comparison(individual_vars, portfolio_var, tickers, weights):
    """
    Compare individual stock VaRs to portfolio VaR.

    Shows the benefits of diversification.
    """

    # create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # prepare data
    labels = list(tickers) + ['Portfolio']
    values = [individual_vars[t] for t in tickers] + [portfolio_var]
    colors_list = ['lightcoral'] * len(tickers) + ['green']

    # create bar chart
    bars = ax.bar(labels, values, color=colors_list, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    
    # add value labels and weights
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        
        # value label
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2%}',
                ha='center', va='bottom' if value < 0 else 'top',
                fontsize=11, fontweight='bold')
        
        # weight label for individual stocks
        if i < len(tickers):
            ax.text(bar.get_x() + bar.get_width()/2., 0,
                    f'({weights[i]:.0%})',
                    ha='center', va='top',
                    fontsize=9, style='italic')
            
    # add horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='-')

    # calculate and show diversification benefit
    avg_individual = np.mean([individual_vars[t] for t in tickers])
    benefit = avg_individual - portfolio_var

    # add arrow showing diversification benefit
    ax.annotate('', xy=(len(tickers), portfolio_var), 
                xytext=(len(tickers), avg_individual),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    
    ax.text(len(tickers) + 0.3, (avg_individual + portfolio_var) / 2,
            f'Diversification\nBenefit:\n{benefit:.2%}',
            fontsize=10, fontweight='bold', color='blue',
            va='center')
    
    # labels and title
    ax.set_ylabel('VaR (Potential Loss)', fontsize=12, fontweight='bold')
    ax.set_title('Portfolio Diversification Effect\nIndividual Stocks vs Portfolio VaR',
                 fontsize=14, fontweight='bold')
    
    # format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'{y:.1%}'))
    
    # add grid
    ax.grid(axis='y', alpha=0.3)
    
    # add textbox with summary interpretation
    textstr = f'Portfolio Risk: {portfolio_var:.2%}\n'
    textstr += f'Avg Individual Risk: {avg_individual:.2%}\n'
    textstr += f'Risk Reduction: {abs(benefit):.2%}\n\n'
    
    # calculate average individual VaR
    avg_individual = np.mean([individual_vars[t] for t in tickers])

    # compare portfolio to average
    if abs(portfolio_var) < abs(avg_individual):
        # portfolio is less risky (diversification success)
        textstr += f'The portfolio is LESS risky than\nthe average of individual stocks!'
    elif abs(portfolio_var) > abs(avg_individual):
        # portfolio is more risky (high correlation/ diversification failure)
        textstr += f'The portfolio is MORE risky than\nthe average of individual stocks!\n(High correlation detected)'
    else:
        # exactly equal risk
        textstr += f'The portfolio has same risk to\nthe average of individual stocks'

    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig
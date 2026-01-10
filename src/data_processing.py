# Take yfinance data and process it in order to pass to calculations file.

import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.data = None
        self.returns = None

    """Fetch historical data from yfinance."""
    def fetch_data(self, ticker, start_date, end_date):
        
        try:

            # check for single ticker, if so, convert to list
            if isinstance(tickers, str):
                tickers = [tickers]
                single_ticker = True
            else:
                single_ticker = False

            # fetch all tickers in one API call
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # throw error if no data found
            if data.empty:
                raise ValueError("No data found. Please check the ticker symbol(s): {tickers}")
            
            # if single ticker, display as Series
            if single_ticker:
                if 'Close' in data.columns:
                    prices = data['Close']
                else:
                    prices = data

            # display as DataFrame for multiple tickers
            else:
                if 'Close' in data.columns.levels[0]:
                    prices = data['Close']
                else:
                    prices = data

            # drop columns with all NaN values
            prices = prices.dropna(axis=1, how='all')

            # store historical stock data in object
            self.data = prices
        
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    '''Calculate daily returns from price data'''
    def calculate_returns(self, price_data):

        # Calculate percent change and drop NaN values from first row
        returns = price_data.pct_change().dropna()

        self.returns = returns

        if isinstance(returns, pd.DataFrame):
            print(f"Calculated {len(returns)} daily returns for {len(returns.columns)} stocks.")
        else:
            print(f"Calculated {len(returns)} daily returns.")

        return returns

    '''Get portfolio data and returns given tickers, weights, and date range'''
    def get_portfolio_data(self, tickers, weights, start_date, end_date):

        # throw error if weights do not add up to 1
        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0, currently at {sum(weights)}")
        
        # confirm number of tickers matches number of weights
        if len(tickers) != len(weights):
            raise ValueError("Number of tickers must match number of weights")
        
        # Fetch data
        prices = self.fetch_data(tickers, start_date, end_date)

        # throw error if data fetch failed
        if prices is None:
            raise ValueError("Failed to fetch portfolio data")
        
        # Fetch daily returns for all stocks
        individual_returns = self.calculate_returns(prices)

        # Calculate portfolio returns as weighted sum of individual returns
        # Multiply each stock's returns by its weight, then sum across columns
        portfolio_returns = (individual_returns * weights).sum(axis=1)

        self.returns = portfolio_returns
        self.individual_returns = individual_returns

        print(f"Portfolio data ready: {len(portfolio_returns)} observations")
        print(f"Portfolio composition: {dict(zip(tickers, weights))}")
        
        return individual_returns, portfolio_returns
    

    '''Get statistics of the current data'''
    def get_data_summary(self):

        if self.returns is None:
            raise ValueError("No returns data available.")
        
        summary = {
            "Number of observations": len(self.returns),
            "Mean daily return": f"{self.returns.mean():.4f}",
            "Std deviation": f"{self.returns.std():.4f}",
            "Min daily return": f"{self.returns.min():.4f}",
            "Max daily return": f"{self.returns.max():.4f}",
            "Start date": str(self.returns.index.min().date()),
            "End date": str(self.returns.index.max().date())
        }

        return summary
    
    '''Get summary statistics for all stocks in the portfolio'''
    def get_portfolio_summary(self, tickers, weights):

        if not hasattr(self, 'individual_returns') or self.individual_returns is None:
            return {"error": "No individual stock data available. Run get_portfolio_data() first."}
        
        summary_data = []

        for i, ticker in enumerate(tickers):
            if ticker in self.individual_returns.columns:
                returns = self.individual_returns[ticker]

                summary_data.append({
                    "Ticker": ticker,
                    "Weight": f"{weights[i]:.4%}",
                    "Mean daily return": f"{returns.mean():.4f}",
                    "Std deviation": f"{returns.std():.4f}",
                    "Min daily return": f"{returns.min():.4f}",
                    "Max daily return": f"{returns.max():.4f}"
                })

        summary_df = pd.DataFrame(summary_data)
        return summary_df
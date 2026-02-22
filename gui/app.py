"""
Streamlit Web Application for VaR Calculator
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# add src directory to sys.path for module imports
sys.path.append('.')

# import modules
from data_handler import DataHandler

from calculations import (
    historical_var, parametric_var, monte_carlo_var,
    conditional_var, calculate_var_dollars,
    compare_var_methods
)

from visualizations import (
    plot_distribution_with_var, plot_var_comparison,
    plot_portfolio_comparison, plot_cvar_comparison,
    plot_rolling_var, create_var_dashboard
)

# set page configuration
st.set_page_config(
    page_title="VaR Calculator",
    layout="wide",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# header
st.markdown('<div class="main-header">📊 VaR Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Professional Risk Analysis for Stocks & Portfolios</div>', unsafe_allow_html=True)

st.markdown("---")

# sidebar for user inputs
with st.sidebar:
    st.header("Configuration")

    # type of analysis
    st.subheader("1. Analysis Type")
    analysis_type = st.radio(
        "Choose analysis:",
        ["Single Stock", "Portfolio"],
        help="Single stock analyzes one ticker. Portfolio analyzes multiple stocks with weights."
    )

    st.markdown("---")

    # tickers inputs
    st.subheader("2. Stock selection(s)")

    if analysis_type == "Single Stock":
        ticker = st.text_input(
            "Enter Stock Ticker:",
            value="AAPL",
            help="Example: AAPL for Apple, MSFT for Microsoft, GOOGL for Google)."
        ).upper()

        tickers = [ticker]
        weights = [1.0]

    else: # Portfolio 
        # number of stocks
        num_stocks = st.number_input(
            "Number of Stocks:",
            min_value=2,
            max_value=10,
            value=3,
            help="How many stocks are in your portfolio?"
        )

        tickers = []
        weights = []

        st.write("Enter tickers and weights:**")

        # dynamic inputs for each stock
        for i in range(num_stocks):
            col1, col2 = st.columns([2,1])

            with col1:
                ticker_input = st.text_input(
                    f"Stock {i+1}:",
                    value=["AAPL", "MSFT", "GOOGL", "TSLA", 
                           "NVDA", "META", "AMZN", "NFLX", "AMD", "INTC"][i] 
                           if i < 10 else "",
                           key=f"ticker_{i}"
                ).upper()
                tickers.append(ticker_input)

            with col2:
                weight_input = st.number_input(
                    "Weight:",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0/num_stocks,
                    step=0.05,
                    format="%.2f",
                    key=f"weight_{i}"
                )
                weights.append(weight_input)

        # normalize weights
        total_weight = sum(weights)
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total_weight:.2f}. Should equal 1.0")
        else:
            st.success(f"✅ Weights sum to {total_weight:.2f}")

    st.markdown("---")

    # portfolio value
    st.subheader("3.Portfolio Value")
    portfolio_value = st.number_input(
        "Total Portfolio Value ($):",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000,
        help="Used to convert VaR percentage to dollar amounts"
    )

    st.markdown("---")

    # date range
    st.subheader("4. Date Range")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=datetime.now() - timedelta(days=365),
            help="Beginning of historical data"
        )

    with col2:
        end_date = st.date_input(
            "End Date:",
            value=datetime.now(),
            help="End of historical data"
        )

    st.markdown("---")

    # confidence level
    st.subheader("5. Confidence Level")
    confidence_level = st.slider(
        "Select Confidence Level:",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.0f%%",
        help="95% means you're 95% confident losses won't exceed VaR"
    )

    st.info(f"📊 **{confidence_level:.0%} confidence** means VaR will be exceeded on ~{(1-confidence_level)*100:.0f}% of days")

    st.markdown("---")

    # calculate button
    calculate_button = st.button("🚀 Calculate VaR", type="primary", use_container_width=True)

    # about section
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This calculator computes Value at Risk (VaR) using:
    - **Historical VaR**: Actual past data
    - **Parametric VaR**: Normal distribution
    - **Monte Carlo VaR**: Simulations
    - **CVaR**: Average tail loss
    """)

    # result dashboard

if calculate_button:

    # validation
    if not all(tickers):
        st.error("Please enter all stock tickers.")
        st.stop()

    if analysis_type == "Portfolio" and abs(sum(weights) - 1.0) > 0.01:
        st.error("Weights must sum to 1. Please adjust.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    # loading spinner
    with st.spinner("Fetching data and calculating VaR..."):

        try:
            handler = DataHandler()

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # fetch data

            # single stock analysis
            if analysis_type == "Single Stock":
                prices = handler.fetch_stock_data(tickers[0], start_str, end_str)

                if prices is None or len(prices) == 0:
                    st.error(f"No data found for {tickers[0]}. Please check the ticker and date range.")
                    st.stop()

                returns = handler.calculate_returns(prices)
                individual_returns = None
                portfolio_returns = returns

            # portfolio analysis
            else:
                individual_returns, portfolio_returns = handler.get_portfolio_data(tickers, weights, start_str, end_str)

                if individual_returns is None or portfolio_returns is None:
                    st.error("Error fetching portfolio data. Please check tickers, weights, and date range.")
                    st.stop()

            #  ---- calculate VaR metrics

            # stock/portfolio VaR
            hist_var = historical_var(portfolio_returns, confidence_level)
            param_var = parametric_var(portfolio_returns, confidence_level)
            mc_var = monte_carlo_var(portfolio_returns, confidence_level, 10000)
            cvar = conditional_var(portfolio_returns, hist_var, confidence_level, var_threshold=hist_var)

            # convert to dollars
            hist_var_dollars = calculate_var_dollars(hist_var, portfolio_value)
            cvar_dollars = calculate_var_dollars(cvar, portfolio_value)

            # individual stock VaR (for portfolio analysis)
            individual_vars = {}
            if analysis_type == "Portfolio":
                for ticker in tickers:
                    individual_vars[ticker] = historical_var(individual_returns[ticker], confidence_level)
                        
            # ---- display results

            st.success("✅Analysis complete!")

            # summary metrics
            st.header("📊 Summary Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Historical VaR",
                          f"{hist_var:.2%}", 
                          delta=f"${abs(hist_var_dollars):,.0f} loss",
                          delta_color="inverse"
                          )
                
            with col2:
                st.metric("Parametric VaR",
                          f"{param_var:.2%}",
                          delta=f"{(param_var - hist_var)*100:.2f} vs Hist",
                          delta_color="inverse"
                          )
                
            with col3:
                st.metric("Monte Carlo VaR",
                          f"{mc_var:.2%}",
                          delta=f"${abs(mc_var - hist_var):,.0f} vs Hist",
                          delta_color="inverse"
                          )
                
            with col4:
                st.metric("CVaR (Expected Shortfall)",
                          f"{cvar:.2%}",
                          delta=f"${abs(cvar_dollars):,.0f} CVaR/Avg Tail Loss",
                          delta_color="inverse"
                          )
            st.markdown("---")

            # ---- data summary

            st.header("📈 Data Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Dataset information")
                summary = handler.get_data_summary()

                summary_df = pd.DataFrame({
                    'Metric': list(summary.keys()),
                    'Value': list(summary.values())
                })
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

            with col2:
                st.subheader("Portfolio Composition")

                if analysis_type == "Portfolio":
                    composition_df = pd.DataFrame({
                        'Ticker': tickers,
                        'Weight': [f"{w:.2%}" for w in weights],
                        'Dollar Amount': [f"${portfolio_value * w:,.0f}" for w in weights]
                    })
                    st.dataframe(composition_df, hide_index=True, use_container_width=True)
                else:
                    st.info(f"**Single Stock Analysis**\n\nTicker: {tickers[0]}\n\nPortfolio Value: ${portfolio_value:,.0f}")

            st.markdown("---")

            # ---- detailed analysis

            st.header("🔍 Detailed Analysis")

            #VaR Comparison Table
            st.subheader("VaR Method Comparison")

            var_comparison_df = pd.DataFrame({
                'Method': ['Historical', 'Parametric', 'Monte Carlo', 'Conditional VaR (CVaR)'],
                'VaR (%)': [f"{hist_var:.2%}", f"{param_var:.2%}", f"{mc_var:.2%}", f"{cvar:.2%}"],
                'Dollar Loss': [
                    f"${abs(hist_var_dollars):,.0f}",
                    f"${abs(calculate_var_dollars(param_var, portfolio_value)):,.0f}",
                    f"${abs(calculate_var_dollars(mc_var, portfolio_value)):,.0f}",
                    f"${abs(cvar_dollars):,.0f}"
                ],
                'Interpretation': [
                    f"{confidence_level:.0%} of days won't exceed this loss",
                    "Based on normal distribution assumption",
                    "Based on 10,000 simulations",
                    f"Average loss on worst {(1-confidence_level):.0%} of days"
                ]
            })

            st.dataframe(var_comparison_df, hide_index=True, use_container_width=True)

            # individual stock VaRs (for portfolio)
            if analysis_type == "Portfolio":
                st.markdown("---")
                st.subheader("Individual Stock VaRs")
                
                individual_df = pd.DataFrame({
                    'Ticker': tickers,
                    'Weight': [f"{w:.1%}" for w in weights],
                    'VaR (%)': [f"{individual_vars[t]:.2%}" for t in tickers],
                    'Dollar Loss': [f"${abs(calculate_var_dollars(individual_vars[t], portfolio_value * weights[i])):,.0f}" 
                                   for i, t in enumerate(tickers)]
                })
                
                st.dataframe(individual_df, hide_index=True, use_container_width=True)

                # benefit of diversification?
                avg_individual = np.mean([individual_vars[t] for t in tickers])
                diversification_benefit = avg_individual - hist_var
                
                if abs(diversification_benefit) > 0.001:
                    if diversification_benefit > 0:
                        st.success(f"""
                        ✅ **Diversification Benefit Detected!**
                        
                        - Average Individual Stock VaR: {avg_individual:.2%}
                        - Portfolio VaR: {hist_var:.2%}
                        - **Risk Reduction: {diversification_benefit:.2%}**
                        
                        Your portfolio is **less risky** than the average of individual stocks due to diversification!
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ **High Correlation Detected**
                        
                        - Average Individual Stock VaR: {avg_individual:.2%}
                        - Portfolio VaR: {hist_var:.2%}
                        - **Additional Risk: {abs(diversification_benefit):.2%}**
                        
                        Your portfolio is **more risky** than expected, suggesting high correlation between stocks.
                        """)
            
            st.markdown("---")

            # ---- visualizations

            st.header("📊 Visualizations")

            # create table for different visualizations
            if analysis_type == "Portfolio":
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Distribution",
                    "📊 Method Comparison",
                    "🎯 Portfolio vs Stocks",
                    "⚠️ VaR vs CVaR",
                    "🎨 Dashboard"
                ])
            else:
                tab1, tab2, tab4, tab5 = st.tabs([
                    "📈 Distribution",
                    "📊 Method Comparison",
                    "⚠️ VaR vs CVaR",
                    "🎨 Dashboard"
                ])
            
            with tab1:
                st.subheader("Return Distribution with VaR Threshold")
                fig1 = plot_distribution_with_var(portfolio_returns, hist_var, confidence_level, cvar, "Historical")
                st.pyplot(fig1)
                plt.close(fig1)
                
                st.info("""
                **How to read this chart:**
                - The histogram shows all historical returns
                - Red bars highlight the worst 5% of days (the "tail")
                - Red line = VaR threshold (95% of days are better than this)
                - Orange line = CVaR (average of the worst days)
                """)
            
            with tab2:
                st.subheader("Comparison of VaR Methods")
                var_results = {
                    'Historical': hist_var,
                    'Parametric': param_var,
                    'Monte Carlo': mc_var
                }
                fig2 = plot_var_comparison(var_results, confidence_level)
                st.pyplot(fig2)
                plt.close(fig2)
                
                st.info("""
                **How to read this chart:**
                - All three methods calculate VaR differently
                - Similar values = returns are relatively normal
                - Large differences = non-normal returns or extreme events
                """)
            
            if analysis_type == "Portfolio":
                with tab3:
                    st.subheader("Portfolio vs Individual Stock Risk")
                    fig3 = plot_portfolio_comparison(individual_vars, hist_var, tickers, weights)
                    st.pyplot(fig3)
                    plt.close(fig3)
                    
                    st.info("""
                    **How to read this chart:**
                    - Individual bars show each stock's VaR
                    - Green bar shows portfolio VaR
                    - If portfolio is lower = diversification is working!
                    """)
            
            with tab4:
                st.subheader("VaR vs CVaR - Understanding Tail Risk")
                fig4 = plot_cvar_comparison(hist_var, cvar, confidence_level)
                st.pyplot(fig4)
                plt.close(fig4)
                
                st.info("""
                **How to read this chart:**
                - VaR = threshold (95% of days won't be worse)
                - CVaR = average when things DO go bad
                - The gap shows "tail risk" beyond VaR
                """)
            
            with tab5:
                st.subheader("Comprehensive Dashboard")
                
                if analysis_type == "Portfolio":
                    fig5 = create_var_dashboard(
                        portfolio_returns,
                        tickers=tickers,
                        weights=weights,
                        individual_vars=individual_vars,
                        portfolio_var=hist_var,
                        confidence_level=confidence_level
                    )
                else:
                    fig5 = create_var_dashboard(
                        portfolio_returns,
                        confidence_level=confidence_level
                    )
                
                st.pyplot(fig5)
                plt.close(fig5)
                
                st.info("**All-in-one view** showing distribution, method comparison, portfolio analysis, and tail risk.")
            
            st.markdown("---")

            # ---- interpretation and insights

            st.header("💡 Interpretation & Insights")

            # risk level assessment
            abs_var = abs(hist_var)
            
            if abs_var < 0.015:  # < 1.5%
                risk_level = "LOW"
                color_class = "success-card"
                emoji = "✅"
            elif abs_var < 0.025:  # 1.5% - 2.5%
                risk_level = "MODERATE"
                color_class = "warning-card"
                emoji = "⚠️"
            else:  # > 2.5%
                risk_level = "HIGH"
                color_class = "danger-card"
                emoji = "🚨"
            
            st.markdown(f"""
            <div class="{color_class}">
            <h3>{emoji} Risk Level: {risk_level}</h3>
            <p><strong>Historical VaR ({confidence_level:.0%} confidence): {hist_var:.2%}</strong></p>
            <p>Expected maximum loss: ${abs(hist_var_dollars):,.0f}</p>
            <p>This means: <strong>{confidence_level:.0%} of days, you won't lose more than {abs(hist_var):.2%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("###")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Key Takeaways")
                
                st.markdown(f"""
                - **VaR ({confidence_level:.0%})**: {hist_var:.2%} or ${abs(hist_var_dollars):,.0f}
                - **CVaR**: {cvar:.2%} or ${abs(cvar_dollars):,.0f}
                - **Tail Risk**: {abs(cvar - hist_var):.2%} additional loss in worst scenarios
                - **Data Period**: {(end_date - start_date).days} days ({len(portfolio_returns)} observations)
                """)
                
                if analysis_type == "Portfolio":
                    if diversification_benefit > 0:
                        st.markdown(f"- **Diversification**: Working! {diversification_benefit:.2%} risk reduction")
                    else:
                        st.markdown(f"- **Diversification**: Limited due to high correlation")
            
            with col2:
                st.subheader("🎯 Recommendations")
                
                if risk_level == "HIGH":
                    st.markdown("""
                    - Consider reducing position sizes
                    - Diversify into less correlated assets
                    - Review risk tolerance
                    - Consider hedging strategies
                    """)
                elif risk_level == "MODERATE":
                    st.markdown("""
                    - Monitor positions regularly
                    - Maintain current diversification
                    - Review during high volatility periods
                    - Consider stop-loss levels
                    """)
                else:
                    st.markdown("""
                    - Risk level is manageable
                    - Continue current strategy
                    - Monitor for changes in volatility
                    - Rebalance periodically
                    """)
        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)

else:
    # Initial state - show instructions
    st.info("""
    👈 **Get Started:**
    
    1. Choose analysis type (Single Stock or Portfolio)
    2. Enter ticker symbols
    3. Select date range
    4. Choose confidence level
    5. Click "Calculate VaR"
    
    The app will fetch historical data, calculate VaR using multiple methods, and display comprehensive results with visualizations.
    """)
    
    st.markdown("---")

# ---- footer

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>VaR Calculator</strong> | Built with Streamlit & Python</p>
    <p>Educational tool for demonstrating financial risk analysis concepts</p>
    <p style='font-size: 0.8rem;'>Data sourced from Yahoo Finance | Past performance does not guarantee future results</p>
</div>
""", unsafe_allow_html=True)
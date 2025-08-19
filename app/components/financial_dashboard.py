"""Financial dashboard components for BuffettBot."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.financial_analyzer import BuffettFinancialAnalyzer

def create_stock_input():
    """Create stock symbol input interface."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_symbol = st.text_input(
            "Enter Stock Symbol:",
            value="AAPL",
            placeholder="e.g., AAPL, MSFT, GOOGL",
            help="Enter any publicly traded stock symbol"
        )
    
    with col2:
        analyze_button = st.button("📊 Analyze", use_container_width=True)
    
    return stock_symbol.upper() if analyze_button else None

def display_company_header(analyzer: BuffettFinancialAnalyzer):
    """Display company information header."""
    company_info = analyzer.get_company_info()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"## 🏢 {company_info.get('name', analyzer.symbol)}")
        st.markdown(f"**Symbol:** {analyzer.symbol}")
        st.markdown(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {company_info.get('industry', 'N/A')}")
    
    with col2:
        market_cap = company_info.get('market_cap', 0)
        if market_cap > 0:
            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        
        employees = company_info.get('employees', 'N/A')
        if employees != 'N/A':
            st.metric("Employees", f"{employees:,}")
    
    with col3:
        # Get Buffett score
        passed, total = analyzer.get_buffett_score()
        score_pct = (passed/total)*100 if total > 0 else 0
        
        score_color = "🟢" if score_pct >= 80 else "🟡" if score_pct >= 60 else "🟠" if score_pct >= 40 else "🔴"
        st.metric("Buffett Score", f"{score_color} {passed}/{total}", f"{score_pct:.1f}%")

def display_buffett_ratios(analyzer: BuffettFinancialAnalyzer):
    """Display Warren Buffett's financial ratios analysis."""
    st.subheader("📊 Warren Buffett's Financial Analysis")
    
    analysis = analyzer.analyze_investment_quality()
    ratios = analysis['ratios']
    
    # Overall assessment
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {analysis['quality']}")
        st.markdown(f"**Score:** {analysis['score']} ({analysis['percentage']:.1f}%)")
        st.markdown(f"**Assessment:** {analysis['recommendation']}")
    
    with col2:
        # Create a gauge chart for the score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = analysis['percentage'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Buffett Score"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed ratios
    st.markdown("### 📋 Detailed Financial Ratios")
    
    # Group ratios by category
    income_ratios = {}
    balance_ratios = {}
    cashflow_ratios = {}
    
    for name, data in ratios.items():
        if 'Margin' in name and name != 'CapEx Margin':
            income_ratios[name] = data
        elif 'Debt' in name or 'Cash' in name:
            balance_ratios[name] = data
        else:
            cashflow_ratios[name] = data
    
    # Display ratios in tabs
    tab1, tab2, tab3 = st.tabs(["📈 Income Statement", "⚖️ Balance Sheet", "💰 Cash Flow"])
    
    with tab1:
        display_ratio_table(income_ratios, "Income Statement Ratios")
    
    with tab2:
        display_ratio_table(balance_ratios, "Balance Sheet Ratios")
    
    with tab3:
        display_ratio_table(cashflow_ratios, "Cash Flow Ratios")

def display_ratio_table(ratios: Dict, title: str):
    """Display a table of financial ratios."""
    if not ratios:
        st.info("No ratios available for this category")
        return
    
    # Create DataFrame for display
    ratio_data = []
    for name, data in ratios.items():
        status = "✅ Pass" if data['passes'] else "❌ Fail"
        value_display = f"{data['value']:.1%}" if isinstance(data['value'], (int, float)) else str(data['value'])
        
        ratio_data.append({
            'Metric': name,
            'Value': value_display,
            'Buffett Rule': data['rule'],
            'Status': status,
            'Description': data['description']
        })
    
    df = pd.DataFrame(ratio_data)
    
    # Style the dataframe
    def highlight_status(val):
        color = '#90EE90' if '✅' in val else '#FFB6C1'  # Light green for pass, light red for fail
        return f'background-color: {color}'
    
    styled_df = df.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def display_financial_statements(analyzer: BuffettFinancialAnalyzer):
    """Display detailed financial statements."""
    st.subheader("📋 Financial Statements")
    
    statements = analyzer.get_financial_statements()
    
    if not statements:
        st.error("Unable to load financial statements")
        return
    
    # Create tabs for each statement
    tab1, tab2, tab3 = st.tabs(["📈 Income Statement", "⚖️ Balance Sheet", "💰 Cash Flow"])
    
    with tab1:
        st.markdown("#### Income Statement")
        if 'income_statement' in statements:
            st.dataframe(statements['income_statement'], use_container_width=True)
        else:
            st.error("Income statement not available")
    
    with tab2:
        st.markdown("#### Balance Sheet")
        if 'balance_sheet' in statements:
            st.dataframe(statements['balance_sheet'], use_container_width=True)
        else:
            st.error("Balance sheet not available")
    
    with tab3:
        st.markdown("#### Cash Flow Statement")
        if 'cash_flow' in statements:
            st.dataframe(statements['cash_flow'], use_container_width=True)
        else:
            st.error("Cash flow statement not available")

def create_ratio_visualization(ratios: Dict):
    """Create visualization of Buffett ratios."""
    if not ratios:
        return
    
    # Prepare data for visualization
    ratio_names = []
    values = []
    statuses = []
    
    for name, data in ratios.items():
        if isinstance(data['value'], (int, float)):
            ratio_names.append(name)
            values.append(data['value'] * 100)  # Convert to percentage
            statuses.append("Pass" if data['passes'] else "Fail")
    
    if not ratio_names:
        return
    
    # Create bar chart
    fig = px.bar(
        x=ratio_names,
        y=values,
        color=statuses,
        color_discrete_map={"Pass": "green", "Fail": "red"},
        title="Warren Buffett's Financial Ratios Analysis",
        labels={'x': 'Financial Ratios', 'y': 'Value (%)'}
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_investment_summary(analyzer: BuffettFinancialAnalyzer):
    """Display investment summary and recommendations."""
    analysis = analyzer.analyze_investment_quality()
    
    st.subheader("🎯 Investment Summary")
    
    # Create summary cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Quality", analysis['quality'])
    
    with col2:
        st.metric("Criteria Met", analysis['score'])
    
    with col3:
        st.metric("Success Rate", f"{analysis['percentage']:.1f}%")
    
    with col4:
        # Simple buy/hold/sell recommendation
        if analysis['percentage'] >= 80:
            recommendation = "🟢 BUY"
        elif analysis['percentage'] >= 60:
            recommendation = "🟡 HOLD"
        else:
            recommendation = "🔴 AVOID"
        st.metric("Recommendation", recommendation)
    
    # Detailed assessment
    st.markdown("### 💭 Buffett-Style Analysis")
    st.info(analysis['recommendation'])
    
    # Key strengths and weaknesses
    strengths = []
    weaknesses = []
    
    for name, data in analysis['ratios'].items():
        if data['passes']:
            strengths.append(f"✅ **{name}**: {data['description']}")
        else:
            weaknesses.append(f"❌ **{name}**: {data['description']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💪 Strengths")
        for strength in strengths[:5]:  # Show top 5
            st.markdown(strength)
    
    with col2:
        st.markdown("#### ⚠️ Areas of Concern")
        for weakness in weaknesses[:5]:  # Show top 5
            st.markdown(weakness)

def create_comparison_tool():
    """Create a tool to compare multiple stocks."""
    st.subheader("⚖️ Stock Comparison Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock1 = st.text_input("Stock 1:", value="AAPL", key="compare1")
    with col2:
        stock2 = st.text_input("Stock 2:", value="MSFT", key="compare2")
    with col3:
        stock3 = st.text_input("Stock 3:", value="GOOGL", key="compare3")
    
    if st.button("Compare Stocks"):
        symbols = [s for s in [stock1, stock2, stock3] if s.strip()]
        
        comparison_data = []
        for symbol in symbols:
            analyzer = BuffettFinancialAnalyzer()
            if analyzer.load_stock_data(symbol):
                analysis = analyzer.analyze_investment_quality()
                passed, total = analyzer.get_buffett_score()
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Company': analyzer.get_company_info().get('name', symbol),
                    'Buffett Score': f"{passed}/{total}",
                    'Score %': f"{analysis['percentage']:.1f}%",
                    'Quality': analysis['quality'],
                    'Recommendation': analysis['recommendation']
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Unable to analyze any of the provided symbols")
# ADD THESE IMPORTS to the top of your existing app.py
import yfinance as yf

# ADD THIS FUNCTION after your existing functions in app.py
def financial_analysis_tab():
    """Add financial analysis as a new tab."""
    st.header("📊 Warren Buffett Stock Analyzer")
    st.markdown("*Analyze any stock using Buffett's investment criteria*")
    
    # Stock input
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol:", value="AAPL", placeholder="e.g., AAPL, MSFT, GOOGL")
    with col2:
        analyze = st.button("📈 Analyze", use_container_width=True)
    
    if analyze and symbol:
        symbol = symbol.upper()
        
        try:
            with st.spinner(f"📊 Analyzing {symbol}..."):
                # Get stock data
                stock = yf.Ticker(symbol)
                financials = stock.financials
                balance_sheet = stock.balancesheet
                info = stock.info
                
                if financials.empty:
                    st.error(f"❌ No data found for {symbol}")
                    return
                
                # Company header
                company_name = info.get('longName', symbol)
                st.markdown(f"### 🏢 {company_name}")
                
                # Get latest year data
                latest_year = financials.columns[0]
                
                # Calculate key Buffett ratios
                gross_profit = financials.loc['Gross Profit', latest_year]
                total_revenue = financials.loc['Total Revenue', latest_year]
                net_income = financials.loc['Net Income', latest_year]
                
                # 1. Gross Margin (≥40%)
                gross_margin = gross_profit / total_revenue
                
                # 2. Net Profit Margin (≥20%)  
                net_margin = net_income / total_revenue
                
                # 3. SG&A Margin (≤30%) - if available
                sga_margin = None
                try:
                    sga = financials.loc['Selling General And Administration', latest_year]
                    sga_margin = sga / gross_profit
                except:
                    pass
                
                # 4. R&D Margin (≤30%) - if available
                rd_margin = None
                try:
                    rd = financials.loc['Research And Development', latest_year]
                    rd_margin = rd / gross_profit
                except:
                    pass
                
                # Display results
                st.markdown("#### 💰 Warren Buffett's Key Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gross Margin", f"{gross_margin:.1%}")
                    status = "✅ Pass" if gross_margin >= 0.40 else "❌ Fail"
                    st.write(f"{status} (Rule: ≥40%)")
                
                with col2:
                    st.metric("Net Profit Margin", f"{net_margin:.1%}")
                    status = "✅ Pass" if net_margin >= 0.20 else "❌ Fail"
                    st.write(f"{status} (Rule: ≥20%)")
                
                with col3:
                    if sga_margin is not None:
                        st.metric("SG&A Margin", f"{sga_margin:.1%}")
                        status = "✅ Pass" if sga_margin <= 0.30 else "❌ Fail"
                        st.write(f"{status} (Rule: ≤30%)")
                    else:
                        st.metric("SG&A Margin", "N/A")
                
                with col4:
                    if rd_margin is not None:
                        st.metric("R&D Margin", f"{rd_margin:.1%}")
                        status = "✅ Pass" if rd_margin <= 0.30 else "❌ Fail"
                        st.write(f"{status} (Rule: ≤30%)")
                    else:
                        st.metric("R&D Margin", "N/A")
                
                # Calculate Buffett Score
                criteria_met = 0
                total_criteria = 2  # Always have gross and net margin
                
                if gross_margin >= 0.40:
                    criteria_met += 1
                if net_margin >= 0.20:
                    criteria_met += 1
                
                if sga_margin is not None:
                    total_criteria += 1
                    if sga_margin <= 0.30:
                        criteria_met += 1
                
                if rd_margin is not None:
                    total_criteria += 1
                    if rd_margin <= 0.30:
                        criteria_met += 1
                
                score_pct = (criteria_met / total_criteria) * 100
                
                # Investment recommendation
                st.markdown("---")
                st.markdown("#### 🎯 Investment Assessment")
                
                if score_pct >= 75:
                    st.success(f"🟢 **EXCELLENT** - {criteria_met}/{total_criteria} criteria met ({score_pct:.0f}%)")
                    st.markdown("✅ Strong Buffett-style investment candidate")
                elif score_pct >= 50:
                    st.warning(f"🟡 **GOOD** - {criteria_met}/{total_criteria} criteria met ({score_pct:.0f}%)")
                    st.markdown("⚠️ Meets some Buffett criteria, investigate further")
                else:
                    st.error(f"🔴 **POOR** - {criteria_met}/{total_criteria} criteria met ({score_pct:.0f}%)")
                    st.markdown("❌ Does not meet Buffett's investment standards")
                
                # Show financial statements
                with st.expander("📋 View Financial Statements"):
                    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                    
                    with tab1:
                        st.dataframe(financials.head(15), use_container_width=True)
                    with tab2:
                        if not balance_sheet.empty:
                            st.dataframe(balance_sheet.head(15), use_container_width=True)
                        else:
                            st.info("Balance sheet data not available")
                    with tab3:
                        cashflow = stock.cashflow
                        if not cashflow.empty:
                            st.dataframe(cashflow.head(15), use_container_width=True)
                        else:
                            st.info("Cash flow data not available")
                
        except Exception as e:
            st.error(f"❌ Error analyzing {symbol}: {e}")
    
    # FINANCIAL PAGE
    elif st.session_state.current_page == "Financial":
        financial_analysis_tab()

# ADD THIS TO THE END of your existing app.py main() function
# Replace the existing main() function with this enhanced version:

def main():
    """Enhanced main function with tabs."""
    
    # Your existing initialization code here...
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Chat"
    
    # Your existing system loading code...
    if "system" not in st.session_state:
        st.session_state.system = load_buffett_system()
    
    df, embedder, index, tokenizer, model, device = st.session_state.system
    
    # Navigation in sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Warren_Buffett_KU_Visit.jpg/256px-Warren_Buffett_KU_Visit.jpg", 
                caption="Warren Buffett", width=200)
        
        st.markdown("---")
        
        # Simple page selection
        page = st.radio("Choose Feature:", ["💬 Chat with Warren", "📊 Analyze Stocks"])
        
        if page == "💬 Chat with Warren":
            st.session_state.current_page = "Chat"
        else:
            st.session_state.current_page = "Financial"
    
    # Show appropriate page
    if st.session_state.current_page == "Chat":
        # YOUR EXISTING CHAT CODE GOES HERE
        # (Keep all your existing chat functionality)
        pass
    
    elif st.session_state.current_page == "Financial":
        financial_analysis_tab()

# Keep your existing functions and just modify main()
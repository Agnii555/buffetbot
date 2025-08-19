"""
BuffettBot - Fixed version with proper torch imports
Replace your app/streamlit_app.py with this code
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Page config
if 'page_configured' not in st.session_state:
    st.set_page_config(
        page_title="BuffettBot - Investment Advisor & Analyzer",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_configured = True

# Try imports with proper error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    OPTION_MENU_AVAILABLE = True
except ImportError:
    OPTION_MENU_AVAILABLE = False

@st.cache_resource(show_spinner=False)
def load_chatbot_system():
    """Load chatbot with proper torch handling."""
    
    if not all([TORCH_AVAILABLE, EMBEDDINGS_AVAILABLE, FAISS_AVAILABLE, TRANSFORMERS_AVAILABLE]):
        missing = []
        if not TORCH_AVAILABLE:
            missing.append("torch")
        if not EMBEDDINGS_AVAILABLE:
            missing.append("sentence-transformers")
        if not FAISS_AVAILABLE:
            missing.append("faiss-cpu")
        if not TRANSFORMERS_AVAILABLE:
            missing.append("transformers")
        
        st.error(f"❌ Missing dependencies: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")
        return None
    
    try:
        # Load data
        dataset_paths = [
            "Dataset_Warren_Buffet_Clean.csv",
            "data/Dataset_Warren_Buffet_Clean.csv",
            Path(__file__).parent.parent / "data" / "Dataset_Warren_Buffet_Clean.csv"
        ]
        
        df = None
        for path in dataset_paths:
            try:
                df = pd.read_csv(path)
                break
            except:
                continue
        
        if df is None:
            st.error("❌ Dataset not found. Please place Dataset_Warren_Buffet_Clean.csv in the project directory.")
            return None
        
        # Load models with proper torch usage
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        answers = df['answer'].tolist()
        embeddings = embedder.encode(answers, convert_to_numpy=True, show_progress_bar=False)
        
        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        
        # Load generation model with proper torch device handling
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        return {
            'df': df,
            'embedder': embedder,
            'index': index,
            'tokenizer': tokenizer,
            'model': model,
            'device': device,
            'loaded': True
        }
        
    except Exception as e:
        st.error(f"❌ Error loading system: {e}")
        return None

def search_and_generate(query, system, k=3, max_length=200):
    """Search knowledge and generate response with proper torch handling."""
    try:
        # Search phase
        embedder = system['embedder']
        index = system['index']
        df = system['df']
        
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding.astype(np.float32), k)
        
        # Get context
        contexts = []
        sources = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:
                row = df.iloc[idx]
                contexts.append(row['answer'])
                sources.append({
                    'question': row['question'],
                    'answer': row['answer'],
                    'category': row.get('refined_category', 'General'),
                    'relevance': 1 / (1 + distance)
                })
        
        context = "\n\n".join(contexts)
        
        # Generation phase with proper torch usage
        tokenizer = system['tokenizer']
        model = system['model']
        device = system['device']
        
        prompt = f"Context from Warren Buffett's teachings:\n{context[:800]}\n\nQuestion: {query}\n\nWarren Buffett's response:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, 
                          truncation=True, padding=True).to(device)
        
        # Generate with proper torch context
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip(), context, sources
        
    except Exception as e:
        error_msg = f"I apologize, but I'm having trouble generating a response. Error: {str(e)}"
        return error_msg, "", []

def chat_page():
    """Chat interface with fixed torch handling."""
    st.title("💬 BuffettBot Chat")
    st.markdown("*Ask Warren Buffett about investment philosophy and strategies*")
    
    # Load system
    system = load_chatbot_system()
    if not system:
        return
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat settings
    with st.sidebar:
        st.header("⚙️ Chat Settings")
        context_docs = st.slider("Context Documents", 1, 5, 3, key="chat_context")
        response_length = st.slider("Response Length", 50, 400, 200, key="chat_length")
        show_context = st.checkbox("Show Knowledge Sources", True, key="show_context")
        
        st.header("📊 Chat Statistics")
        st.metric("Knowledge Base", f"{len(system['df']):,} Q&A pairs")
        st.metric("Categories", f"{len(system['df']['refined_category'].unique())}")
        st.metric("Messages This Session", len(st.session_state.chat_history))
        
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Sample questions if no history
    if not st.session_state.chat_history:
        st.markdown("### 💡 Popular Investment Questions")
        st.markdown("*Click any question to get started:*")
        
        sample_questions = [
            "Why is margin of safety important in investing?",
            "How do you evaluate company management quality?",
            "What makes a good long-term investment?",
            "When should I sell a stock?",
            "How do you handle market volatility?",
            "What's your approach to diversification?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            if cols[i % 2].button(question, key=f"sample_q_{i}"):
                st.session_state.pending_question = question
                st.rerun()
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(message["user"])
        
        with st.chat_message("assistant", avatar="💰"):
            st.caption("💎 BuffettBot")
            st.write(message["response"])
            
            if show_context and message.get("sources"):
                with st.expander("📚 Knowledge Sources", expanded=False):
                    for j, source in enumerate(message["sources"][:2]):
                        st.markdown(f"**📖 Source {j+1}** (Category: {source['category']}, Relevance: {source['relevance']:.2f})")
                        st.markdown(f"**Original Q:** {source['question']}")
                        st.markdown(f"**Buffett's Teaching:** {source['answer'][:250]}...")
                        if j < len(message["sources"]) - 1:
                            st.markdown("---")
    
    # Handle input
    user_input = None
    
    # Handle pending sample question
    if hasattr(st.session_state, 'pending_question'):
        user_input = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
    
    # Chat input
    if not user_input:
        user_input = st.chat_input("💭 Ask Warren Buffett about investing...")
    
    # Process input
    if user_input and user_input.strip():
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response
        with st.chat_message("assistant", avatar="💰"):
            st.caption("💎 BuffettBot")
            
            with st.spinner("💭 Consulting Warren's investment wisdom..."):
                # Generate response with error handling
                response, context, sources = search_and_generate(
                    user_input, system, k=context_docs, max_length=response_length
                )
                
                # Display response
                st.write(response)
                
                # Show sources if enabled and available
                if show_context and sources:
                    with st.expander("📚 Knowledge Sources", expanded=False):
                        for i, source in enumerate(sources[:2]):
                            st.markdown(f"**📖 Source {i+1}** (Category: {source['category']}, Relevance: {source['relevance']:.2f})")
                            st.markdown(f"**Original Q:** {source['question']}")
                            st.markdown(f"**Buffett's Teaching:** {source['answer'][:250]}...")
                            if i < len(sources) - 1:
                                st.markdown("---")
                
                # Add to history
                st.session_state.chat_history.append({
                    "user": user_input,
                    "response": response,
                    "context": context,
                    "sources": sources
                })

def financial_page():
    """Financial analysis page."""
    st.title("📊 Warren Buffett Stock Analyzer")
    st.markdown("*Analyze any stock using Buffett's proven investment criteria*")
    
    # Check yfinance availability
    if not YFINANCE_AVAILABLE:
        st.error("❌ Financial analysis requires yfinance")
        st.code("pip install yfinance")
        st.info("Install yfinance to unlock stock analysis features")
        return
    
    # Stock input
    col1, col2 = st.columns([4, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol:", value="AAPL", 
                              placeholder="e.g., AAPL, MSFT, GOOGL, TSLA")
    with col2:
        if st.button("📈 Analyze", use_container_width=True):
            if symbol.strip():
                st.session_state.analyze_symbol = symbol.upper().strip()
                st.rerun()
    
    # Analysis settings
    with st.sidebar:
        st.header("📊 Analysis Settings")
        show_full_data = st.checkbox("Show Complete Financial Statements", False)
        show_explanations = st.checkbox("Show Buffett's Logic", True)
    
    # Perform analysis
    if hasattr(st.session_state, 'analyze_symbol'):
        symbol = st.session_state.analyze_symbol
        
        try:
            with st.spinner(f"📊 Analyzing {symbol} using Warren Buffett's criteria..."):
                # Load stock data
                stock = yf.Ticker(symbol)
                financials = stock.financials
                balance_sheet = stock.balancesheet
                info = stock.info
                
                if financials.empty:
                    st.error(f"❌ No financial data available for {symbol}")
                    st.info("Please verify the stock symbol is correct and the company is publicly traded.")
                    return
                
                # Company information
                company_name = info.get('longName', symbol)
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                market_cap = info.get('marketCap', 0)
                
                # Company header
                st.markdown(f"## 🏢 {company_name}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Symbol", symbol)
                with col2:
                    st.metric("Sector", sector)
                with col3:
                    if len(industry) > 15:
                        display_industry = industry[:15] + "..."
                    else:
                        display_industry = industry
                    st.metric("Industry", display_industry)
                with col4:
                    if market_cap > 0:
                        if market_cap >= 1e12:
                            cap_display = f"${market_cap/1e12:.1f}T"
                        elif market_cap >= 1e9:
                            cap_display = f"${market_cap/1e9:.1f}B"
                        else:
                            cap_display = f"${market_cap/1e6:.0f}M"
                        st.metric("Market Cap", cap_display)
                    else:
                        st.metric("Market Cap", "N/A")
                
                st.markdown("---")
                
                # Calculate Warren Buffett's key ratios
                latest_year = financials.columns[0]
                
                # Core financial metrics
                gross_profit = financials.loc['Gross Profit', latest_year]
                total_revenue = financials.loc['Total Revenue', latest_year]
                net_income = financials.loc['Net Income', latest_year]
                
                # Calculate ratios
                gross_margin = gross_profit / total_revenue
                net_margin = net_income / total_revenue
                
                # Optional ratios (may not be available for all companies)
                sga_margin = None
                rd_margin = None
                depreciation_margin = None
                
                try:
                    sga = financials.loc['Selling General And Administration', latest_year]
                    if pd.notna(sga) and sga > 0:
                        sga_margin = sga / gross_profit
                except:
                    pass
                
                try:
                    rd = financials.loc['Research And Development', latest_year]
                    if pd.notna(rd) and rd > 0:
                        rd_margin = rd / gross_profit
                except:
                    pass
                
                try:
                    depreciation = financials.loc['Reconciled Depreciation', latest_year]
                    if pd.notna(depreciation) and depreciation > 0:
                        depreciation_margin = depreciation / gross_profit
                except:
                    pass
                
                # Display Warren Buffett Analysis
                st.subheader("💰 Warren Buffett's Investment Criteria")
                
                # Track score
                criteria_met = 0
                total_criteria = 2  # Always have gross and net margin
                
                # Core ratios display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gross Margin", f"{gross_margin:.1%}")
                    if gross_margin >= 0.40:
                        st.success("✅ Excellent (≥40%)")
                        criteria_met += 1
                    elif gross_margin >= 0.30:
                        st.warning("⚠️ Decent (30-40%)")
                    else:
                        st.error("❌ Poor (<30%)")
                    
                    if show_explanations:
                        st.caption("🎯 Buffett's Logic: High gross margins indicate the company has pricing power and isn't competing solely on price.")
                
                with col2:
                    st.metric("Net Profit Margin", f"{net_margin:.1%}")
                    if net_margin >= 0.20:
                        st.success("✅ Excellent (≥20%)")
                        criteria_met += 1
                    elif net_margin >= 0.10:
                        st.warning("⚠️ Decent (10-20%)")
                    else:
                        st.error("❌ Poor (<10%)")
                    
                    if show_explanations:
                        st.caption("🎯 Buffett's Logic: Great companies convert 20% or more of their revenue into net profit.")
                
                with col3:
                    if sga_margin is not None:
                        st.metric("SG&A Expense Margin", f"{sga_margin:.1%}")
                        total_criteria += 1
                        if sga_margin <= 0.30:
                            st.success("✅ Efficient (≤30%)")
                            criteria_met += 1
                        else:
                            st.error("❌ High (>30%)")
                        
                        if show_explanations:
                            st.caption("🎯 Buffett's Logic: Wide-moat companies don't need to spend heavily on overhead to operate.")
                    else:
                        st.metric("SG&A Expense Margin", "N/A")
                        st.info("Data not available")
                
                with col4:
                    if rd_margin is not None:
                        st.metric("R&D Expense Margin", f"{rd_margin:.1%}")
                        total_criteria += 1
                        if rd_margin <= 0.30:
                            st.success("✅ Disciplined (≤30%)")
                            criteria_met += 1
                        else:
                            st.error("❌ High (>30%)")
                        
                        if show_explanations:
                            st.caption("🎯 Buffett's Logic: R&D expenses don't always create lasting value for shareholders.")
                    else:
                        st.metric("R&D Expense Margin", "N/A")
                        st.info("Data not available")
                
                # Additional metrics if available
                if depreciation_margin is not None:
                    st.markdown("#### 🏭 Asset Efficiency")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Depreciation Margin", f"{depreciation_margin:.1%}")
                        total_criteria += 1
                        if depreciation_margin <= 0.10:
                            st.success("✅ Asset-light (≤10%)")
                            criteria_met += 1
                        else:
                            st.error("❌ Asset-heavy (>10%)")
                        
                        if show_explanations:
                            st.caption("🎯 Buffett's Logic: Great businesses don't need lots of depreciating assets to maintain their advantage.")
                
                # Calculate and display overall score
                st.markdown("---")
                score_percentage = (criteria_met / total_criteria) * 100 if total_criteria > 0 else 0
                
                st.subheader("🎯 Overall Investment Assessment")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if score_percentage >= 80:
                        st.success(f"🟢 **EXCELLENT INVESTMENT** - Meets {criteria_met}/{total_criteria} Buffett criteria")
                        st.markdown("✅ This company strongly aligns with Warren Buffett's investment philosophy and represents a high-quality investment opportunity.")
                        recommendation = "🟢 STRONG BUY"
                    elif score_percentage >= 60:
                        st.warning(f"🟡 **GOOD INVESTMENT** - Meets {criteria_met}/{total_criteria} Buffett criteria")
                        st.markdown("⚠️ This company meets many of Buffett's criteria but has some areas that need attention. Consider for investment with additional research.")
                        recommendation = "🟡 BUY WITH CAUTION"
                    elif score_percentage >= 40:
                        st.warning(f"🟠 **FAIR INVESTMENT** - Meets {criteria_met}/{total_criteria} Buffett criteria")
                        st.markdown("🤔 Mixed results. This company has some positive qualities but significant concerns that require deeper investigation.")
                        recommendation = "🟠 INVESTIGATE FURTHER"
                    else:
                        st.error(f"🔴 **POOR INVESTMENT** - Only meets {criteria_met}/{total_criteria} Buffett criteria")
                        st.markdown("❌ This company does not meet Warren Buffett's investment standards and should likely be avoided.")
                        recommendation = "🔴 AVOID"
                
                with col2:
                    st.metric("Buffett Score", f"{criteria_met}/{total_criteria}", f"{score_percentage:.0f}%")
                    st.markdown("**Recommendation:**")
                    st.markdown(recommendation)
                
                # Show full financial data if requested
                if show_full_data:
                    st.markdown("---")
                    st.subheader("📋 Complete Financial Statements")
                    
                    tab1, tab2, tab3 = st.tabs(["📈 Income Statement", "⚖️ Balance Sheet", "💰 Cash Flow"])
                    
                    with tab1:
                        st.markdown("*Most recent 4 years of income statement data:*")
                        st.dataframe(financials, use_container_width=True)
                    
                    with tab2:
                        st.markdown("*Most recent 4 years of balance sheet data:*")
                        st.dataframe(balance_sheet, use_container_width=True)
                    
                    with tab3:
                        cashflow = stock.cashflow
                        st.markdown("*Most recent 4 years of cash flow data:*")
                        st.dataframe(cashflow, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error analyzing {symbol}: {e}")
            st.info("Please verify the stock symbol is correct and try again.")
    
    else:
        # Welcome screen for financial analysis
        st.markdown("""
        ### 👋 Welcome to Warren Buffett's Stock Analyzer!
        
        **Enter any stock symbol above to get a comprehensive analysis using Warren Buffett's time-tested investment criteria.**
        
        #### 🎯 What You'll Discover:
        - **📊 Buffett Score** - Overall investment quality rating (0-100%)
        - **📈 Key Financial Ratios** - All of Warren Buffett's critical metrics
        - **🎯 Investment Recommendation** - Clear guidance on whether to buy, hold, or avoid
        - **📋 Complete Financial Statements** - Full access to company financial data
        
        #### 💰 Warren Buffett's Proven Investment Criteria:
        
        **📈 Profitability Indicators:**
        - **Gross Margin ≥ 40%** - Indicates pricing power and competitive advantage
        - **Net Profit Margin ≥ 20%** - Shows strong profitability and operational efficiency
        
        **💼 Operational Efficiency:**
        - **SG&A Expenses ≤ 30% of Gross Profit** - Low overhead indicates a wide economic moat
        - **R&D Expenses ≤ 30% of Gross Profit** - Disciplined capital allocation
        
        **🏭 Asset Management:**
        - **Low Depreciation** - Asset-light business models are preferred
        - **Strong Cash Generation** - Financial independence and flexibility
        
        #### 🔥 Try Analyzing These Popular Stocks:
        """)
        
        # Popular stock analysis buttons
        popular_stocks = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corp."),
            ("GOOGL", "Alphabet Inc."),
            ("BRK-B", "Berkshire Hathaway"),
            ("KO", "Coca-Cola Co."),
            ("JPM", "JPMorgan Chase")
        ]
        
        cols = st.columns(3)
        for i, (ticker, name) in enumerate(popular_stocks):
            if cols[i % 3].button(f"📊 {ticker}\n*{name}*", key=f"popular_stock_{ticker}"):
                st.session_state.analyze_symbol = ticker
                st.rerun()

def main():
    """Main application with enhanced navigation."""
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Chat"
    
    # Sidebar with navigation
    with st.sidebar:
        # Warren Buffett image
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Warren_Buffett_KU_Visit.jpg/256px-Warren_Buffett_KU_Visit.jpg", 
                caption="Warren Buffett - The Oracle of Omaha", width=200)
        
        st.markdown("---")
        
        # Navigation menu
        if OPTION_MENU_AVAILABLE:
            # Use fancy menu if available
            selected = option_menu(
                menu_title="BuffettBot",
                options=["💬 Chat Assistant", "📊 Financial Analyzer"],
                icons=["chat-dots", "graph-up"],
                menu_icon="currency-dollar",
                default_index=0 if st.session_state.current_page == "Chat" else 1,
                styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )
            
            # Update current page based on selection
            if "Chat" in selected:
                st.session_state.current_page = "Chat"
            elif "Financial" in selected:
                st.session_state.current_page = "Financial"
        
        else:
            # Fallback to simple buttons
            st.header("🧭 Navigation")
            
            if st.button("💬 Chat Assistant", use_container_width=True, 
                        type="primary" if st.session_state.current_page == "Chat" else "secondary"):
                st.session_state.current_page = "Chat"
                st.rerun()
            
            if st.button("📊 Financial Analyzer", use_container_width=True,
                        type="primary" if st.session_state.current_page == "Financial" else "secondary"):
                st.session_state.current_page = "Financial"
                st.rerun()
        
        st.markdown("---")
        
        # System status
        st.header("🔧 System Status")
        
        # Check all systems
        chat_system = load_chatbot_system()
        if chat_system and chat_system.get('loaded'):
            st.success("💬 Chat: Ready")
            st.caption(f"📚 {len(chat_system['df']):,} Q&A pairs loaded")
        else:
            st.error("💬 Chat: Error")
        
        if YFINANCE_AVAILABLE:
            st.success("📊 Financial: Ready")
            st.caption("🏢 Real-time stock analysis")
        else:
            st.error("📊 Financial: Missing yfinance")
    
    # Route to appropriate page
    if st.session_state.current_page == "Chat":
        chat_page()
    elif st.session_state.current_page == "Financial":
        financial_page()
    
    # Footer
    st.markdown("---")
    
    footer_cols = st.columns(4)
    
    with footer_cols[0]:
        st.markdown("**🎯 BuffettBot Features**")
        st.markdown("• AI chat with Warren's wisdom")
        st.markdown("• Real-time stock analysis")
        st.markdown("• Investment recommendations")
    
    with footer_cols[1]:
        st.markdown("**📚 Knowledge Base**")
        st.markdown("• 4,996+ Q&A pairs")
        st.markdown("• Investment philosophy")
        st.markdown("• Financial ratio analysis")
    
    with footer_cols[2]:
        st.markdown("**🏆 Technology**")
        st.markdown("• FLAN-T5 AI model")
        st.markdown("• Semantic search")
        st.markdown("• Real-time financial data")
    
    with footer_cols[3]:
        st.markdown("**⚠️ Important Disclaimer**")
        st.markdown("• Educational purposes only")
        st.markdown("• Not financial advice")
        st.markdown("• Consult qualified advisors")

if __name__ == "__main__":
    main()
"""Chat interface components for BuffettBot Streamlit app."""

import streamlit as st
from typing import List, Dict
import time

def display_chat_message(message: Dict, show_context: bool = True):
    """Display a single chat message with styling."""
    
    # User message
    with st.chat_message("user"):
        st.write(message["user"])
    
    # Bot response
    with st.chat_message("assistant"):
        # Show model name badge
        model_badge = "🤖 Base FLAN-T5" if "Base" in message.get("model", "") else "💎 BuffettBot"
        st.caption(model_badge)
        
        # Show response
        st.write(message["bot"])
        
        # Show context in expander if requested
        if show_context and "context" in message:
            with st.expander("📚 Retrieved Context", expanded=False):
                st.text_area("Context used for this response:", 
                           value=message["context"], 
                           height=150, 
                           disabled=True)

def display_chat_history(history: List[Dict]):
    """Display the entire chat history."""
    if not history:
        st.info("👋 Welcome! Ask me anything about Warren Buffett's investment philosophy.")
        return
    
    for message in history:
        display_chat_message(message)

def create_sample_questions():
    """Create sample question buttons."""
    st.subheader("💡 Sample Questions")
    
    sample_questions = [
        "Why is having a margin of safety important?",
        "How do you evaluate company management?",
        "What makes a business have a moat?",
        "When should I sell a stock?",
        "What's your view on diversification?",
        "How do you think about market volatility?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        if col.button(question, key=f"sample_{i}"):
            return question
    
    return None

def create_input_form():
    """Create the input form for new questions."""
    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask BuffettBot:",
                placeholder="Type your investment question here...",
                label_visibility="collapsed"
            )
        
        with col2:
            submitted = st.form_submit_button("Send 📤", use_container_width=True)
        
        return user_input if submitted else None

def create_model_selector():
    """Create model selection interface."""
    st.subheader("🎛️ Model Settings")
    
    model_choice = st.radio(
        "Choose AI Model:",
        ("Base FLAN-T5", "Fine-Tuned BuffettBot"),
        help="Base model uses general knowledge + context. Fine-tuned model is specialized on Buffett's wisdom."
    )
    
    # Advanced settings in expander
    with st.expander("⚙️ Advanced Settings"):
        top_k = st.slider("Context Documents", 1, 5, 3, 
                         help="Number of relevant documents to retrieve")
        max_length = st.slider("Response Length", 50, 300, 200,
                              help="Maximum length of generated response")
        show_context = st.checkbox("Show Retrieved Context", True,
                                  help="Display the context used for each response")
    
    return {
        "model": model_choice,
        "top_k": top_k,
        "max_length": max_length,
        "show_context": show_context
    }

def display_typing_animation():
    """Show typing animation while generating response."""
    with st.empty():
        for i in range(3):
            st.write("💭 BuffettBot is thinking" + "." * (i + 1))
            time.sleep(0.5)
        st.empty()

def display_stats_sidebar(retriever=None):
    """Display system statistics in sidebar."""
    with st.sidebar:
        st.subheader("📊 System Stats")
        
        if retriever:
            stats = retriever.get_stats()
            st.metric("Knowledge Base", f"{stats.get('total_documents', 0)} documents")
            st.metric("Embedding Dimension", stats.get('embedding_dimension', 'N/A'))
        
        # Session stats
        if "history" in st.session_state:
            st.metric("Questions Asked", len(st.session_state.history))
        
        # Model info
        st.info("🧠 **Models Used:**\n- Embeddings: all-MiniLM-L6-v2\n- Generation: FLAN-T5")

def clear_chat_history():
    """Clear the chat history."""
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()
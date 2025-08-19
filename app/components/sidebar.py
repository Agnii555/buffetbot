"""Sidebar components for BuffettBot Streamlit app."""

import streamlit as st
from typing import Dict, List

def create_sidebar_header():
    """Create the sidebar header with Warren Buffett image."""
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Warren_Buffett_KU_Visit.jpg/256px-Warren_Buffett_KU_Visit.jpg",
            caption="Warren Buffett - The Oracle of Omaha",
            width=200
        )
        
        st.markdown("---")

def create_model_settings():
    """Create model selection and settings in sidebar."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.radio(
            "Choose AI Model:",
            ["Base FLAN-T5", "Fine-Tuned BuffettBot"],
            help="Base model uses general knowledge. Fine-tuned is specialized on Buffett's wisdom."
        )
        
        # Retrieval settings
        st.subheader("🔍 Retrieval Settings")
        top_k = st.slider("Documents to Retrieve", 1, 5, 3,
                         help="Number of relevant context documents")
        
        # Generation settings
        st.subheader("🤖 Generation Settings")
        max_length = st.slider("Max Response Length", 50, 400, 200,
                              help="Maximum length of generated responses")
        temperature = st.slider("Response Creativity", 0.1, 1.0, 0.7,
                               help="Higher values = more creative responses")
        
        # Display settings
        st.subheader("👁️ Display Settings")
        show_context = st.checkbox("Show Retrieved Context", True,
                                  help="Display the context used for responses")
        show_similarity = st.checkbox("Show Similarity Scores", False,
                                     help="Display relevance scores for retrieved content")
        
        return {
            "model_choice": model_choice,
            "top_k": top_k,
            "max_length": max_length,
            "temperature": temperature,
            "show_context": show_context,
            "show_similarity": show_similarity
        }

def display_system_stats(processor=None, retriever=None):
    """Display system statistics in sidebar."""
    with st.sidebar:
        st.header("📊 System Statistics")
        
        if processor:
            stats = processor.get_stats()
            
            st.metric("Knowledge Base Size", f"{stats['total_pairs']:,} Q&A pairs")
            st.metric("Investment Categories", f"{stats['categories']}")
            st.metric("Avg Question Length", f"{stats['avg_question_length']:.0f} chars")
            st.metric("Avg Answer Length", f"{stats['avg_answer_length']:.0f} chars")
        
        if retriever:
            retriever_stats = retriever.get_stats()
            st.metric("Search Index Size", f"{retriever_stats.get('total_documents', 0)} docs")
            st.metric("Embedding Dimension", retriever_stats.get('embedding_dimension', 'N/A'))
        
        # Session stats
        if "history" in st.session_state:
            st.metric("Questions This Session", len(st.session_state.history))

def display_category_info(processor=None):
    """Display category information in sidebar."""
    if not processor:
        return
    
    with st.sidebar:
        st.header("📋 Knowledge Categories")
        
        stats = processor.get_stats()
        category_dist = stats.get('category_distribution', {})
        
        # Show top categories
        sorted_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories[:8]:  # Show top 8
            percentage = (count / stats['total_pairs']) * 100
            st.write(f"• **{category}**: {count} items ({percentage:.1f}%)")
        
        if len(sorted_categories) > 8:
            remaining = len(sorted_categories) - 8
            st.write(f"... and {remaining} more categories")

def create_action_buttons():
    """Create action buttons in sidebar."""
    with st.sidebar:
        st.header("🔄 Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
        
        with col2:
            export_chat = st.button("📤 Export Chat", use_container_width=True)
        
        # Random wisdom button
        random_wisdom = st.button("🎲 Random Buffett Wisdom", use_container_width=True)
        
        return {
            "clear_chat": clear_chat,
            "export_chat": export_chat,
            "random_wisdom": random_wisdom
        }

def display_random_wisdom(processor):
    """Display random Warren Buffett wisdom."""
    if processor and processor.df is not None:
        random_qa = processor.df.sample(1).iloc[0]
        
        with st.sidebar:
            st.info(f"**💎 Random Buffett Wisdom:**\n\n**Q:** {random_qa['question']}\n\n**A:** {random_qa['answer'][:200]}...")

def create_help_section():
    """Create help and tips section in sidebar."""
    with st.sidebar:
        st.header("💡 Tips & Help")
        
        with st.expander("📚 How to Use BuffettBot"):
            st.markdown("""
            **Getting Started:**
            1. Ask any investment-related question
            2. BuffettBot will search Warren Buffett's teachings
            3. Get contextual, wisdom-based responses
            
            **Best Questions:**
            - Specific investment scenarios
            - Business evaluation questions
            - Market philosophy inquiries
            - Risk management topics
            
            **Features:**
            - View retrieved context sources
            - Adjust response length and creativity
            - Clear chat history anytime
            """)
        
        with st.expander("🎯 Example Questions"):
            st.markdown("""
            • "How do you value a company?"
            • "What red flags do you look for in management?"
            • "How do you handle market downturns?"
            • "What makes a business worth owning forever?"
            • "How do you think about portfolio concentration?"
            """)

def create_feedback_section():
    """Create feedback collection section."""
    with st.sidebar:
        if st.session_state.get("history"):
            st.header("📝 Feedback")
            
            feedback_type = st.radio(
                "Rate the last response:",
                ["👍 Helpful", "👎 Not Helpful", "🤔 Partially Helpful"],
                key="feedback_radio"
            )
            
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
                # Here you could log feedback to a file or database
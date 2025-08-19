"""
Minimal BuffettBot - Save as app.py
Place Dataset_Warren_Buffet_Clean.csv in the same directory
"""

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="BuffettBot", page_icon="💰", layout="wide")

@st.cache_resource
def setup_buffett_bot():
    """Setup the complete BuffettBot system."""
    
    # Load data
    try:
        df = pd.read_csv("Dataset_Warren_Buffet_Clean.csv")
        st.success(f"✅ Loaded {len(df)} Warren Buffett Q&A pairs")
    except:
        st.error("❌ Please place 'Dataset_Warren_Buffet_Clean.csv' in this directory")
        st.stop()
    
    # Load models
    with st.spinner("🤖 Loading AI models..."):
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build search index
        answers = df['answer'].tolist()
        embeddings = embedder.encode(answers, convert_to_numpy=True)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        
        # Load generation model
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        
    st.success("✅ BuffettBot ready!")
    return df, embedder, index, tokenizer, model

def answer_question(query, df, embedder, index, tokenizer, model, k=3):
    """Answer a question using RAG."""
    
    # Retrieve context
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    
    # Get context
    contexts = []
    for idx in indices[0]:
        if idx >= 0:
            contexts.append(df.iloc[idx]['answer'])
    
    context = "\n\n".join(contexts)
    
    # Generate response
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response.strip(), context

def main():
    """Main app."""
    
    # Setup
    if "system" not in st.session_state:
        st.session_state.system = setup_buffett_bot()
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    df, embedder, index, tokenizer, model = st.session_state.system
    
    # Header
    st.title("💰 BuffettBot")
    st.markdown("*Investment wisdom from Warren Buffett*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Stats")
        st.metric("Knowledge Base", f"{len(df):,} Q&A pairs")
        st.metric("Questions Asked", len(st.session_state.history))
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.rerun()
        
        st.header("💡 Sample Questions")
        samples = [
            "Why is margin of safety important?",
            "How do you evaluate management?", 
            "What makes a good investment?",
            "When should I sell a stock?",
            "What's your view on diversification?"
        ]
        
        for i, q in enumerate(samples):
            if st.button(q, key=f"sample_{i}"):
                st.session_state.sample_q = q
                st.rerun()
    
    # Chat history
    for msg in st.session_state.history:
        with st.chat_message("user"):
            st.write(msg["user"])
        with st.chat_message("assistant"):
            st.write(msg["bot"])
            with st.expander("📚 Context"):
                st.text_area("Retrieved context:", msg["context"], height=100, disabled=True)
    
    # Handle sample question
    user_input = None
    if hasattr(st.session_state, 'sample_q'):
        user_input = st.session_state.sample_q
        delattr(st.session_state, 'sample_q')
    
    # Chat input
    if not user_input:
        user_input = st.chat_input("Ask about Warren Buffett's investment philosophy...")
    
    # Process input
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("💭 Consulting Warren's wisdom..."):
                response, context = answer_question(user_input, df, embedder, index, tokenizer, model)
                
                st.write(response)
                
                with st.expander("📚 Context"):
                    st.text_area("Retrieved context:", context, height=100, disabled=True)
                
                # Add to history
                st.session_state.history.append({
                    "user": user_input,
                    "bot": response,
                    "context": context
                })

if __name__ == "__main__":
    main()
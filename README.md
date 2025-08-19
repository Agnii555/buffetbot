# BuffettBot - Warren Buffett Investment Advisor

🤖 An AI-powered chatbot that provides investment wisdom based on Warren Buffett's teachings and philosophy.

## 🚀 Features

- **💬 Interactive Chat**: Natural conversation interface for investment questions
- **🔍 Semantic Search**: AI-powered search through 4,996+ Warren Buffett Q&A pairs
- **🧠 Contextual Responses**: RAG (Retrieval-Augmented Generation) for accurate answers
- **📚 Source Transparency**: View the exact knowledge used for each response
- **📊 Investment Categories**: Organized wisdom across multiple investment topics
- **🎛️ Customizable**: Adjust response length, context depth, and model settings

## 🏗️ Architecture

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 for semantic search
- **Vector Database**: FAISS for fast similarity search
- **Language Model**: Google FLAN-T5 (base + optional fine-tuned version)
- **Frontend**: Streamlit for interactive web interface
- **Data**: 4,996+ curated Warren Buffett Q&A pairs

## 📋 Quick Setup

### Prerequisites
- Python 3.8+
- 2GB+ RAM
- Internet connection (for initial model downloads)

### Installation

1. **Clone and setup:**
   ```bash
   git clone <repository>
   cd buffett-bot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your data:**
   ```bash
   # Place Dataset_Warren_Buffet_Clean.csv in the data/ folder
   cp Dataset_Warren_Buffet_Clean.csv data/
   ```

4. **Run the application:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## 🎯 Usage Examples

Ask BuffettBot questions like:

- **Value Investing**: "Why is margin of safety important in investing?"
- **Business Analysis**: "How do you evaluate company management quality?"
- **Market Philosophy**: "What's your view on market timing vs time in market?"
- **Risk Management**: "How do you think about portfolio diversification?"
- **Company Evaluation**: "What makes a business worth owning forever?"
- **Investment Strategy**: "When should I sell a stock?"

## 🔧 Advanced Features

### Model Comparison
- Compare responses from base FLAN-T5 vs fine-tuned BuffettBot
- Switch between models in real-time
- See performance differences

### Fine-Tuning (Optional)
```bash
# Train a specialized model on Buffett's teachings (takes 30-60 minutes)
python scripts/train_model.py
```

### Custom Configuration
- Adjust response length and creativity
- Control number of context documents
- Toggle context visibility
- Export chat conversations

## 📁 Project Structure

```
buffett-bot/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── Dataset_Warren_Buffet_Clean.csv
├── models/
│   ├── __init__.py
│   ├── embeddings.py      # Sentence transformer wrapper
│   ├── retriever.py       # FAISS semantic search
│   └── generator.py       # FLAN-T5 response generation
├── utils/
│   ├── __init__.py
│   ├── config.py          # Project configuration
│   └── data_processor.py  # Dataset handling
├── app/
│   ├── __init__.py
│   ├── streamlit_app.py   # Main application
│   └── components/
│       ├── __init__.py
│       ├── chat_interface.py  # Chat UI components
│       └── sidebar.py         # Sidebar components
├── scripts/
│   ├── setup_models.py    # Download and cache models
│   ├── build_index.py     # Build FAISS search index
│   └── train_model.py     # Fine-tune on Buffett data
└── saved_models/          # Cached models and indices
    ├── embeddings/
    ├── fine_tuned/
    └── faiss_index/
```

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/
```

### Building Search Index
```bash
python scripts/build_index.py
```

### Setting up Models
```bash
python scripts/setup_models.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please respect OpenAI's usage policies and Warren Buffett's intellectual property.

## ⚠️ Disclaimer

**Important**: This chatbot is for educational and informational purposes only. It is not intended to provide financial advice or investment recommendations. Always consult with qualified financial advisors before making investment decisions.

## 🔗 Acknowledgments

- Warren Buffett for his timeless investment wisdom
- Hugging Face for the transformers library
- OpenAI for inspiration in conversational AI
- Streamlit for the excellent web framework
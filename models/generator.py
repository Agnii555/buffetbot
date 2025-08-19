"""Text generation models for BuffettBot."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BuffettGenerator:
    """Text generation using FLAN-T5 models."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", use_gpu: bool = True):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info(f"✅ Loaded model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def generate_response(self, 
                         query: str, 
                         context: str = "",
                         max_length: int = 200,
                         temperature: float = 0.7,
                         do_sample: bool = True) -> str:
        """Generate a response given a query and optional context."""
        
        # Construct prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def batch_generate(self, 
                      queries: List[str], 
                      contexts: List[str] = None,
                      max_length: int = 200) -> List[str]:
        """Generate responses for multiple queries."""
        if contexts is None:
            contexts = [""] * len(queries)
        
        responses = []
        for query, context in zip(queries, contexts):
            response = self.generate_response(query, context, max_length)
            responses.append(response)
        
        return responses
    
    def save_model(self, save_path: Path):
        """Save the model and tokenizer."""
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"✅ Saved model to {save_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save model: {e}")
            raise
    
    @classmethod
    def load_model(cls, model_path: Path, use_gpu: bool = True) -> 'BuffettGenerator':
        """Load a saved model."""
        try:
            generator = cls.__new__(cls)
            generator.model_name = str(model_path)
            generator.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
            
            generator.tokenizer = AutoTokenizer.from_pretrained(model_path)
            generator.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            generator.model.to(generator.device)
            
            logger.info(f"✅ Loaded saved model from {model_path}")
            return generator
        except Exception as e:
            logger.error(f"❌ Failed to load model from {model_path}: {e}")
            raise

class BuffettPipeline:
    """High-level pipeline combining retrieval and generation."""
    
    def __init__(self, retriever, base_generator: BuffettGenerator, 
                 fine_tuned_generator: Optional[BuffettGenerator] = None):
        self.retriever = retriever
        self.base_generator = base_generator
        self.fine_tuned_generator = fine_tuned_generator
    
    def answer_question(self, 
                       query: str, 
                       use_fine_tuned: bool = False,
                       k_retrieval: int = 3,
                       max_length: int = 200) -> Dict:
        """Answer a question using RAG approach."""
        
        # Retrieve relevant context
        context = self.retriever.retrieve_context(query, k=k_retrieval)
        retrieved_docs = self.retriever.search(query, k=k_retrieval)
        
        # Choose generator
        generator = self.fine_tuned_generator if (use_fine_tuned and self.fine_tuned_generator) else self.base_generator
        model_used = "Fine-tuned BuffettBot" if (use_fine_tuned and self.fine_tuned_generator) else "Base FLAN-T5"
        
        # Generate response
        response = generator.generate_response(query, context, max_length)
        
        return {
            "query": query,
            "response": response,
            "context": context,
            "retrieved_docs": retrieved_docs,
            "model_used": model_used
        }
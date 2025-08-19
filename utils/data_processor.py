"""Data processing utilities for BuffettBot."""

import pandas as pd
from typing import List, Dict, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BuffettDataProcessor:
    """Handles loading and processing of Warren Buffett Q&A data."""
    
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the Warren Buffett dataset."""
        try:
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"✅ Loaded {len(self.df)} Q&A pairs from dataset")
            return self.df
        except FileNotFoundError:
            logger.error(f"❌ Dataset not found at {self.dataset_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def get_questions_and_answers(self) -> Tuple[List[str], List[str]]:
        """Extract questions and answers as separate lists."""
        if self.df is None:
            self.load_data()
        
        questions = self.df['question'].tolist()
        answers = self.df['answer'].tolist()
        
        return questions, answers
    
    def get_qa_pairs(self) -> List[Dict[str, str]]:
        """Get Q&A pairs as list of dictionaries."""
        if self.df is None:
            self.load_data()
            
        qa_pairs = []
        for _, row in self.df.iterrows():
            qa_pairs.append({
                'question': row['question'],
                'answer': row['answer'],
                'category': row.get('refined_category', 'General'),
                'original_category': row.get('category', 'Uncategorized')
            })
        
        return qa_pairs
    
    def get_categories(self) -> List[str]:
        """Get unique categories from the dataset."""
        if self.df is None:
            self.load_data()
            
        categories = self.df['refined_category'].unique().tolist()
        return sorted([cat for cat in categories if pd.notna(cat)])
    
    def filter_by_category(self, category: str) -> pd.DataFrame:
        """Filter dataset by category."""
        if self.df is None:
            self.load_data()
            
        if category == "All Categories":
            return self.df
        
        filtered_df = self.df[self.df['refined_category'] == category]
        logger.info(f"Filtered to {len(filtered_df)} items in category: {category}")
        return filtered_df
    
    def prepare_training_data(self) -> List[Dict[str, str]]:
        """Prepare data for fine-tuning in the format expected by transformers."""
        if self.df is None:
            self.load_data()
            
        training_data = []
        for _, row in self.df.iterrows():
            training_data.append({
                'input_text': f"question: {row['question']}",
                'target_text': row['answer']
            })
        
        return training_data
    
    def get_sample_questions(self, n: int = 5) -> List[str]:
        """Get sample questions for testing."""
        if self.df is None:
            self.load_data()
            
        return self.df['question'].sample(n=n).tolist()
    
    def search_questions(self, keyword: str) -> pd.DataFrame:
        """Search for questions containing specific keywords."""
        if self.df is None:
            self.load_data()
            
        mask = self.df['question'].str.contains(keyword, case=False, na=False)
        results = self.df[mask]
        logger.info(f"Found {len(results)} questions containing '{keyword}'")
        return results
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            self.load_data()
        
        return {
            "total_pairs": len(self.df),
            "categories": len(self.df['refined_category'].unique()),
            "avg_question_length": self.df['question'].str.len().mean(),
            "avg_answer_length": self.df['answer'].str.len().mean(),
            "category_distribution": self.df['refined_category'].value_counts().to_dict()
        }
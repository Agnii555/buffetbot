"""Fine-tuning script for BuffettBot."""

import sys
from pathlib import Path
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import *
from utils.data_processor import BuffettDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuffettTrainer:
    """Fine-tuning trainer for BuffettBot."""
    
    def __init__(self, base_model_name: str = "google/flan-t5-base"):
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        self.training_dataset = None
    
    def load_base_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"📥 Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
        
        logger.info("✅ Base model loaded successfully")
    
    def prepare_dataset(self, dataset_path: Path):
        """Prepare the training dataset."""
        logger.info("📊 Preparing training dataset...")
        
        # Load data
        processor = BuffettDataProcessor(dataset_path)
        training_data = processor.prepare_training_data()
        
        # Convert to Hugging Face Dataset
        dataset_dict = {
            'input_text': [item['input_text'] for item in training_data],
            'target_text': [item['target_text'] for item in training_data]
        }
        
        self.training_dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"✅ Prepared {len(self.training_dataset)} training examples")
    
    def preprocess_function(self, examples):
        """Tokenize the training examples."""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=MAX_CONTEXT_LENGTH,
            truncation=True,
            padding=True
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target_text'],
                max_length=MAX_RESPONSE_LENGTH,
                truncation=True,
                padding=True
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def train(self, output_dir: Path = None):
        """Fine-tune the model."""
        if output_dir is None:
            output_dir = FINE_TUNED_MODEL_PATH
        
        logger.info("🎯 Starting fine-tuning...")
        
        # Tokenize dataset
        tokenized_dataset = self.training_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.training_dataset.column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=0.01,
            logging_dir=str(output_dir / "logs"),
            logging_steps=100,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=False
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train the model
        trainer.train()
        
        # Save the final model
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"✅ Fine-tuning complete! Model saved to {output_dir}")

def main():
    """Main training function."""
    logger.info("🎓 Starting BuffettBot training...")
    
    # Initialize trainer
    trainer = BuffettTrainer(BASE_MODEL)
    
    # Load base model
    trainer.load_base_model()
    
    # Prepare dataset
    trainer.prepare_dataset(DATASET_PATH)
    
    # Train the model
    trainer.train()
    
    logger.info("🎉 Training complete!")

if __name__ == "__main__":
    main()
"""
Data Processor for Fine-tuning
Handles data loading, preprocessing, and formatting
"""

import json
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import AutoTokenizer


class DataProcessor:
    """Process and format data for fine-tuning"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_from_jsonl(self, file_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                data.append(json.loads(line))
        return data
    
    def format_instruction(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Format data for instruction tuning"""
        if input_text:
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""### Instruction:
{instruction}

### Response:
{output}"""
    
    def format_chat(self, messages: List[Dict]) -> str:
        """Format data for chat fine-tuning"""
        formatted = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted += f"### {role}:\n{content}\n"
        return formatted
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize and format examples"""
        texts = []
        for instruction, input_text, output in zip(
            examples["instruction"],
            examples.get("input", [""] * len(examples["instruction"])),
            examples["output"]
        ):
            formatted = self.format_instruction(instruction, input_text, output)
            texts.append(formatted)
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None
        )
        
        # Create labels for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def create_dataset(self, data: List[Dict]) -> Dataset:
        """Create HuggingFace dataset from data"""
        return Dataset.from_list(data)
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset for training"""
        return dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )

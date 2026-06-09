"""
Ragas Evaluator
Evaluate RAG systems using Ragas metrics
"""

import json
import os
from typing import Dict, Any, List, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall
)
from sentence_transformers import SentenceTransformer
from openai import OpenAI


class RagasEvaluator:
    """Evaluate RAG systems using Ragas"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.embeddings = None
        self.llm = None
        
        self._setup_embeddings()
        self._setup_llm()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_embeddings(self):
        """Setup embedding model for Ragas"""
        embeddings_config = self.config["ragas"]["embeddings"]
        
        self.embeddings = SentenceTransformer(
            embeddings_config["model"]
        )
    
    def _setup_llm(self):
        """Setup LLM for Ragas"""
        llm_config = self.config["ragas"]["llm"]
        
        if llm_config["provider"] == "vllm":
            self.llm = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="empty"
            )
    
    def load_dataset(self, data_path: str) -> Dataset:
        """Load evaluation dataset"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def load_ground_truth(self, data_path: str) -> Dataset:
        """Load ground truth dataset"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def evaluate(
        self,
        dataset: Dataset,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run Ragas evaluation"""
        metrics = metrics or self.config["ragas"]["metrics"]
        
        # Map metric names to Ragas metric objects
        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "context_entity_recall": context_entity_recall
        }
        
        selected_metrics = [metric_map[m] for m in metrics if m in metric_map]
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=selected_metrics,
            llm=self.llm,
            embeddings=self.embeddings,
            batch_size=self.config["ragas"]["batch_size"]
        )
        
        return results
    
    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate RAG system with provided data"""
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        return self.evaluate(dataset)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results"""
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert to dict if needed
        if hasattr(results, 'to_pandas'):
            results_dict = results.to_pandas().to_dict(orient='records')
        else:
            results_dict = results
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            report = df.describe().to_string()
        else:
            report = json.dumps(results, indent=2)
        
        return report

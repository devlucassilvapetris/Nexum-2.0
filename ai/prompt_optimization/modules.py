"""
DSPy Modules
Pre-built modules for common tasks
"""

import dspy
from typing import List, Optional


class RAGModule(dspy.Module):
    """Retrieval-Augmented Generation module"""
    
    def __init__(self, num_passages: int = 3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        """Forward pass"""
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(answer=prediction.answer, context=context)


class QAModule(dspy.Module):
    """Question Answering module"""
    
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        """Forward pass"""
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer)


class ClassificationModule(dspy.Module):
    """Text classification module"""
    
    def __init__(self, classes: List[str]):
        super().__init__()
        self.classes = classes
        self.classify = dspy.ChainOfThought(f"text, classes -> class_label")
    
    def forward(self, text: str) -> dspy.Prediction:
        """Forward pass"""
        classes_str = ", ".join(self.classes)
        prediction = self.classify(text=text, classes=classes_str)
        return dspy.Prediction(class_label=prediction.class_label)


class SummarizationModule(dspy.Module):
    """Text summarization module"""
    
    def __init__(self, max_length: int = 150):
        super().__init__()
        self.max_length = max_length
        self.summarize = dspy.ChainOfThought("text -> summary")
    
    def forward(self, text: str) -> dspy.Prediction:
        """Forward pass"""
        prediction = self.summarize(text=text)
        return dspy.Prediction(summary=prediction.summary)


class TranslationModule(dspy.Module):
    """Translation module"""
    
    def __init__(self, target_language: str):
        super().__init__()
        self.target_language = target_language
        self.translate = dspy.ChainOfThought("text, target_language -> translation")
    
    def forward(self, text: str) -> dspy.Prediction:
        """Forward pass"""
        prediction = self.translate(
            text=text,
            target_language=self.target_language
        )
        return dspy.Prediction(translation=prediction.translation)


class CodeGenerationModule(dspy.Module):
    """Code generation module"""
    
    def __init__(self, language: str = "python"):
        super().__init__()
        self.language = language
        self.generate = dspy.ChainOfThought("description, language -> code")
    
    def forward(self, description: str) -> dspy.Prediction:
        """Forward pass"""
        prediction = self.generate(
            description=description,
            language=self.language
        )
        return dspy.Prediction(code=prediction.code)

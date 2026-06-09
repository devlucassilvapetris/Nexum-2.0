"""
RAG Engine
Main RAG implementation with LangChain + vLLM + Instructor
"""

import time
import json
from typing import List, Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from openai import OpenAI
import instructor

from .vector_store import VectorStore
from .schemas import Answer, Document, QueryResponse


class RAGEngine:
    """Retrieval-Augmented Generation engine"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.vector_store = VectorStore(self.config)
        self.vllm_client = None
        self.instructor_client = None
        
        self._setup_vllm_client()
        self._setup_instructor_client()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _setup_vllm_client(self):
        """Setup vLLM client"""
        llm_config = self.config["llm"]
        
        self.vllm_client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="empty"
        )
    
    def _setup_instructor_client(self):
        """Setup Instructor client for structured outputs"""
        if self.config.get("instructor", {}).get("enabled", True):
            self.instructor_client = instructor.patch(
                OpenAI(base_url="http://localhost:8000/v1", api_key="empty")
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store"""
        return self.vector_store.add_documents(documents)
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_rag: bool = True
    ) -> QueryResponse:
        """Query the RAG system"""
        start_time = time.time()
        
        top_k = top_k or self.config["retrieval"]["top_k"]
        
        if use_rag:
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.search(question, top_k)
            
            # Format context
            context = self._format_context(retrieved_docs)
            
            # Generate answer with context
            answer = self._generate_answer_with_context(question, context)
        else:
            # Generate answer without retrieval
            retrieved_docs = []
            answer = self._generate_answer(question)
        
        query_time = time.time() - start_time
        
        # Convert to Document objects
        documents = [
            Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                score=doc["score"]
            )
            for doc in retrieved_docs
        ]
        
        return QueryResponse(
            answer=answer,
            retrieved_documents=documents,
            query_time=query_time
        )
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(
                f"Document {i} (Score: {doc['score']:.4f}):\n{doc['content']}"
            )
        return "\n\n".join(context_parts)
    
    def _generate_answer_with_context(self, question: str, context: str) -> Answer:
        """Generate answer with context using Instructor"""
        if self.instructor_client:
            return self._generate_structured_answer(question, context)
        else:
            return self._generate_text_answer(question, context)
    
    def _generate_structured_answer(self, question: str, context: str) -> Answer:
        """Generate structured answer with Instructor"""
        prompt = f"""
        Based on the following context, answer the question.
        
        Context:
        {context}
        
        Question: {question}
        """
        
        answer = self.instructor_client.chat.completions.create(
            model=self.config["llm"]["model"],
            response_model=Answer,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"]
        )
        
        return answer
    
    def _generate_text_answer(self, question: str, context: str) -> Answer:
        """Generate text answer without structured output"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context."),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nProvide a clear and accurate answer.")
        ])
        
        messages = prompt.format_messages(context=context, question=question)
        
        response = self.vllm_client.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=[{"role": m.type, "content": m.content} for m in messages],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"]
        )
        
        return Answer(
            answer=response.choices[0].message.content,
            sources=[],
            confidence=0.8,
            reasoning="Generated based on retrieved context"
        )
    
    def _generate_answer(self, question: str) -> Answer:
        """Generate answer without context"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "Question: {question}")
        ])
        
        messages = prompt.format_messages(question=question)
        
        response = self.vllm_client.chat.completions.create(
            model=self.config["llm"]["model"],
            messages=[{"role": m.type, "content": m.content} for m in messages],
            temperature=self.config["llm"]["temperature"],
            max_tokens=self.config["llm"]["max_tokens"]
        )
        
        return Answer(
            answer=response.choices[0].message.content,
            sources=[],
            confidence=0.7,
            reasoning="Generated without context"
        )
    
    def batch_query(self, questions: List[str]) -> List[QueryResponse]:
        """Query multiple questions in batch"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results

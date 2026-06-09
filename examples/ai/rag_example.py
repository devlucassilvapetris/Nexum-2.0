"""
Example: RAG Application with LangChain + vLLM + Qdrant + Instructor
"""

from nexum.ai.rag import RAGEngine


def main():
    # Initialize RAG engine
    rag = RAGEngine("config/ai/rag_config.json")
    
    # Add documents
    documents = [
        {
            "id": "doc1",
            "content": "Nexum v2.0 is a sophisticated hybrid operating system combining robotics, computer vision, and AI.",
            "metadata": {"source": "README", "category": "overview"}
        },
        {
            "id": "doc2",
            "content": "The AI layer includes local models, neural networks, and generative AI capabilities.",
            "metadata": {"source": "README", "category": "ai"}
        },
        {
            "id": "doc3",
            "content": "Fine-tuning uses Unsloth for efficient training with PEFT for parameter-efficient fine-tuning.",
            "metadata": {"source": "docs", "category": "training"}
        }
    ]
    
    rag.add_documents(documents)
    
    # Query the RAG system
    question = "What is Nexum v2.0?"
    response = rag.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {response.answer.answer}")
    print(f"Confidence: {response.answer.confidence}")
    print(f"Query time: {response.query_time:.2f}s")
    print(f"Sources: {response.answer.sources}")


if __name__ == "__main__":
    main()

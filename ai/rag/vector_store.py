"""
Vector Store Integration
Support for Qdrant and Milvus
"""

import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue


class VectorStore:
    """Vector store for RAG with Qdrant and Milvus support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = None
        self.qdrant_client = None
        self.milvus_client = None
        
        self._setup_embedding_model()
        self._setup_vector_db()
    
    def _setup_embedding_model(self):
        """Setup sentence transformer for embeddings"""
        embedding_config = self.config["embedding"]
        self.embedding_model = SentenceTransformer(
            embedding_config["model"],
            device=embedding_config.get("device", "cuda")
        )
    
    def _setup_vector_db(self):
        """Setup vector database (Qdrant or Milvus)"""
        if self.config["vector_db"]["type"] == "qdrant":
            self._setup_qdrant()
        elif self.config["milvus"]["enabled"]:
            self._setup_milvus()
    
    def _setup_qdrant(self):
        """Setup Qdrant client"""
        qdrant_config = self.config["vector_db"]
        self.qdrant_client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config["port"]
        )
        
        # Create collection if it doesn't exist
        self._ensure_qdrant_collection()
    
    def _ensure_qdrant_collection(self):
        """Ensure Qdrant collection exists"""
        qdrant_config = self.config["vector_db"]
        collection_name = qdrant_config["collection_name"]
        
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=qdrant_config["vector_size"],
                    distance=Distance.COSINE
                )
            )
    
    def _setup_milvus(self):
        """Setup Milvus client"""
        from pymilvus import connections, Collection, utility
        
        milvus_config = self.config["milvus"]
        
        connections.connect(
            alias="default",
            host=milvus_config["host"],
            port=milvus_config["port"]
        )
        
        # Create collection if it doesn't exist
        self._ensure_milvus_collection()
    
    def _ensure_milvus_collection(self):
        """Ensure Milvus collection exists"""
        from pymilvus import Collection, utility, FieldSchema, CollectionSchema, DataType
        
        milvus_config = self.config["milvus"]
        collection_name = milvus_config["collection_name"]
        
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=milvus_config["dimension"]),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            schema = CollectionSchema(fields, f"{collection_name} collection")
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": milvus_config["metric_type"],
                "index_type": milvus_config["index_type"],
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to vector store"""
        if self.config["vector_db"]["type"] == "qdrant":
            return self._add_documents_qdrant(documents)
        elif self.config["milvus"]["enabled"]:
            return self._add_documents_milvus(documents)
    
    def _add_documents_qdrant(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Qdrant"""
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config["embedding"]["batch_size"],
            show_progress_bar=True
        )
        
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point = PointStruct(
                id=doc.get("id", str(i)),
                vector=embedding.tolist(),
                payload={
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {})
                }
            )
            points.append(point)
        
        self.qdrant_client.upsert(
            collection_name=self.config["vector_db"]["collection_name"],
            points=points
        )
        
        return [point.id for point in points]
    
    def _add_documents_milvus(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to Milvus"""
        from pymilvus import Collection
        
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config["embedding"]["batch_size"],
            show_progress_bar=True
        )
        
        collection = Collection(self.config["milvus"]["collection_name"])
        
        ids = [doc.get("id", str(i)) for i, doc in enumerate(documents)]
        contents = texts
        metadata_list = [json.dumps(doc.get("metadata", {})) for doc in documents]
        
        data = [ids, contents, embeddings.tolist(), metadata_list]
        collection.insert(data)
        collection.flush()
        
        return ids
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.config["vector_db"]["type"] == "qdrant":
            return self._search_qdrant(query, top_k)
        elif self.config["milvus"]["enabled"]:
            return self._search_milvus(query, top_k)
    
    def _search_qdrant(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search in Qdrant"""
        query_embedding = self.embedding_model.encode(query)
        
        results = self.qdrant_client.search(
            collection_name=self.config["vector_db"]["collection_name"],
            query_vector=query_embedding.tolist(),
            limit=top_k,
            score_threshold=self.config["retrieval"].get("score_threshold", 0.0)
        )
        
        documents = []
        for result in results:
            documents.append({
                "id": result.id,
                "content": result.payload["content"],
                "metadata": result.payload["metadata"],
                "score": result.score
            })
        
        return documents
    
    def _search_milvus(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Search in Milvus"""
        from pymilvus import Collection
        
        query_embedding = self.embedding_model.encode(query)
        
        collection = Collection(self.config["milvus"]["collection_name"])
        collection.load()
        
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "metadata"]
        )
        
        documents = []
        for result in results[0]:
            documents.append({
                "id": result.id,
                "content": result.entity.get("content"),
                "metadata": json.loads(result.entity.get("metadata", "{}")),
                "score": result.score
            })
        
        return documents
    
    def delete_collection(self):
        """Delete the entire collection"""
        if self.config["vector_db"]["type"] == "qdrant":
            self.qdrant_client.delete_collection(
                collection_name=self.config["vector_db"]["collection_name"]
            )
        elif self.config["milvus"]["enabled"]:
            from pymilvus import utility
            utility.drop_collection(self.config["milvus"]["collection_name"])

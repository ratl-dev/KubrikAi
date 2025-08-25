"""
Database schema embedding and similarity system.

This module handles schema analysis, embedding generation, and similarity-based
database shortlisting for the intelligent routing system.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
from dataclasses import dataclass, asdict
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


@dataclass
class TableSchema:
    """Represents a database table schema."""
    name: str
    columns: List[Dict[str, Any]]
    primary_keys: List[str]
    foreign_keys: List[Dict[str, str]]
    indexes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    
    def to_text(self) -> str:
        """Convert schema to text representation for embedding."""
        text_parts = [f"Table: {self.name}"]
        
        # Add column information
        for col in self.columns:
            col_text = f"Column {col['name']}: {col['type']}"
            if col.get('nullable', True):
                col_text += " (nullable)"
            text_parts.append(col_text)
        
        # Add key information
        if self.primary_keys:
            text_parts.append(f"Primary keys: {', '.join(self.primary_keys)}")
        
        for fk in self.foreign_keys:
            text_parts.append(f"Foreign key {fk['column']} references {fk['references']}")
        
        return ". ".join(text_parts)


@dataclass
class DatabaseSchema:
    """Represents a complete database schema."""
    db_id: str
    name: str
    domain: str
    tables: List[TableSchema]
    metadata: Dict[str, Any]
    
    def to_text(self) -> str:
        """Convert entire database schema to text for embedding."""
        text_parts = [f"Database: {self.name}", f"Domain: {self.domain}"]
        
        # Add table schemas
        for table in self.tables:
            text_parts.append(table.to_text())
        
        return ". ".join(text_parts)
    
    def get_schema_hash(self) -> str:
        """Generate a hash of the schema for caching."""
        schema_dict = asdict(self)
        schema_str = json.dumps(schema_dict, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()


class SchemaEngine:
    """
    Core schema reasoning and embedding engine.
    
    Handles schema analysis, embedding generation, and similarity-based
    database shortlisting for intelligent routing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the schema engine with embedding model."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(model_name)
            else:
                raise ImportError("SentenceTransformers not available")
        except (ImportError, OSError, Exception) as e:
            # Fallback to mock model for testing
            logger.warning(f"Could not load SentenceTransformer ({e}), using mock embeddings")
            from .mock_embeddings import MockEmbeddingModel
            self.embedding_model = MockEmbeddingModel(model_name)
            
        self.schemas: Dict[str, DatabaseSchema] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(f"Initialized SchemaEngine with model: {model_name}")
    
    def register_database(self, schema: DatabaseSchema) -> None:
        """Register a database schema and generate its embedding."""
        self.schemas[schema.db_id] = schema
        
        # Check cache first
        schema_hash = schema.get_schema_hash()
        if schema_hash in self.embedding_cache:
            self.embeddings[schema.db_id] = self.embedding_cache[schema_hash]
            logger.info(f"Using cached embedding for database {schema.db_id}")
        else:
            # Generate new embedding
            schema_text = schema.to_text()
            embedding = self.embedding_model.encode([schema_text])[0]
            self.embeddings[schema.db_id] = embedding
            self.embedding_cache[schema_hash] = embedding
            logger.info(f"Generated new embedding for database {schema.db_id}")
    
    def get_similar_databases(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Get databases most similar to the query embedding.
        
        Args:
            query_embedding: The query embedding vector
            top_k: Number of top databases to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of (db_id, similarity_score) tuples
        """
        if not self.embeddings:
            logger.warning("No database embeddings available")
            return []
        
        db_ids = list(self.embeddings.keys())
        db_embeddings = np.stack([self.embeddings[db_id] for db_id in db_ids])
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], db_embeddings)[0]
        
        # Filter by threshold and sort
        results = [
            (db_id, float(sim)) 
            for db_id, sim in zip(db_ids, similarities)
            if sim >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a natural language query."""
        return self.embedding_model.encode([query])[0]
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to extract domain hints and requirements.
        
        Returns domain classification and schema requirements.
        """
        query_lower = query.lower()
        
        # Domain classification keywords
        domain_keywords = {
            "ecommerce": ["order", "customer", "product", "payment", "cart", "inventory"],
            "finance": ["account", "transaction", "balance", "payment", "invoice", "budget"],
            "healthcare": ["patient", "doctor", "appointment", "medical", "diagnosis", "treatment"],
            "hr": ["employee", "department", "salary", "performance", "attendance"],
            "marketing": ["campaign", "lead", "conversion", "analytics", "segment"],
            "logistics": ["shipment", "delivery", "warehouse", "tracking", "supplier"]
        }
        
        # Detect likely domains
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Detect data requirements
        requirements = {
            "needs_aggregation": any(word in query_lower for word in ["sum", "count", "average", "max", "min", "total"]),
            "needs_joins": any(word in query_lower for word in ["with", "and", "related", "associated"]),
            "needs_filtering": any(word in query_lower for word in ["where", "filter", "specific", "only"]),
            "needs_sorting": any(word in query_lower for word in ["top", "best", "worst", "order", "sort"]),
            "temporal": any(word in query_lower for word in ["last", "recent", "month", "year", "day", "time"])
        }
        
        return {
            "domain_scores": domain_scores,
            "requirements": requirements,
            "complexity": sum(requirements.values())
        }
    
    def get_schema_compatibility(self, db_id: str, query_analysis: Dict[str, Any]) -> float:
        """
        Calculate how well a database schema matches query requirements.
        
        Args:
            db_id: Database identifier
            query_analysis: Result from analyze_query_intent
        
        Returns:
            Compatibility score between 0 and 1
        """
        if db_id not in self.schemas:
            return 0.0
        
        schema = self.schemas[db_id]
        score = 0.0
        
        # Domain matching
        domain_scores = query_analysis.get("domain_scores", {})
        if domain_scores and schema.domain in domain_scores:
            score += 0.4 * (domain_scores[schema.domain] / max(domain_scores.values()))
        
        # Schema complexity matching
        requirements = query_analysis.get("requirements", {})
        schema_features = {
            "has_foreign_keys": any(table.foreign_keys for table in schema.tables),
            "has_indexes": any(table.indexes for table in schema.tables),
            "has_constraints": any(table.constraints for table in schema.tables),
            "table_count": len(schema.tables)
        }
        
        # Match requirements with schema capabilities
        if requirements.get("needs_joins") and schema_features["has_foreign_keys"]:
            score += 0.2
        
        if requirements.get("needs_filtering") and schema_features["has_indexes"]:
            score += 0.2
        
        if requirements.get("needs_aggregation") and schema_features["table_count"] > 1:
            score += 0.2
        
        return min(score, 1.0)
    
    def get_database_info(self, db_id: str) -> Optional[DatabaseSchema]:
        """Get complete database schema information."""
        return self.schemas.get(db_id)
    
    def list_databases(self) -> List[str]:
        """Get list of all registered database IDs."""
        return list(self.schemas.keys())
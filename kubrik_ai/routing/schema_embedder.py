"""
Schema Embedder - Builds and manages embeddings for database schemas
to enable similarity-based database routing.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
import pickle
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class SchemaMetadata:
    """Metadata for a database schema"""
    database_id: str
    database_type: str  # postgres, mysql, bigquery, etc.
    tables: List[str]
    columns: Dict[str, List[str]]  # table_name -> column_names
    relationships: List[Dict[str, str]]  # foreign key relationships
    indexes: Dict[str, List[str]]  # table_name -> index_names
    description: Optional[str] = None
    domain: Optional[str] = None  # e.g., "e-commerce", "healthcare"


class SchemaEmbedder:
    """
    Builds embeddings for database schemas to enable intelligent routing
    based on semantic similarity between user queries and database schemas.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the schema embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.schema_embeddings: Dict[str, np.ndarray] = {}
        self.schema_metadata: Dict[str, SchemaMetadata] = {}
        
    def extract_schema_features(self, metadata: SchemaMetadata) -> str:
        """
        Extract textual features from schema metadata for embedding.
        
        Args:
            metadata: Schema metadata to extract features from
            
        Returns:
            String representation of schema features
        """
        features = []
        
        # Add database type and description
        features.append(f"Database type: {metadata.database_type}")
        if metadata.description:
            features.append(f"Description: {metadata.description}")
        if metadata.domain:
            features.append(f"Domain: {metadata.domain}")
            
        # Add table information
        table_info = []
        for table in metadata.tables:
            columns = metadata.columns.get(table, [])
            table_desc = f"Table {table} with columns: {', '.join(columns)}"
            table_info.append(table_desc)
        
        features.append("Tables: " + "; ".join(table_info))
        
        # Add relationship information
        if metadata.relationships:
            rel_info = []
            for rel in metadata.relationships:
                rel_desc = f"{rel.get('from_table')}.{rel.get('from_column')} -> {rel.get('to_table')}.{rel.get('to_column')}"
                rel_info.append(rel_desc)
            features.append("Relationships: " + "; ".join(rel_info))
            
        return " ".join(features)
    
    def build_schema_embedding(self, metadata: SchemaMetadata) -> np.ndarray:
        """
        Build embedding for a single database schema.
        
        Args:
            metadata: Schema metadata to build embedding for
            
        Returns:
            Embedding vector for the schema
        """
        schema_text = self.extract_schema_features(metadata)
        embedding = self.model.encode(schema_text)
        return embedding
    
    def add_database_schema(self, metadata: SchemaMetadata) -> None:
        """
        Add a database schema to the embedder.
        
        Args:
            metadata: Schema metadata to add
        """
        logger.info(f"Adding schema for database: {metadata.database_id}")
        
        embedding = self.build_schema_embedding(metadata)
        self.schema_embeddings[metadata.database_id] = embedding
        self.schema_metadata[metadata.database_id] = metadata
        
        logger.info(f"Successfully added schema embedding for {metadata.database_id}")
    
    def find_similar_schemas(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find schemas most similar to a natural language query.
        
        Args:
            query: Natural language query to match against
            top_k: Number of most similar schemas to return
            
        Returns:
            List of (database_id, similarity_score) tuples, sorted by similarity
        """
        if not self.schema_embeddings:
            logger.warning("No schema embeddings available")
            return []
            
        # Encode the query
        query_embedding = self.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for db_id, schema_embedding in self.schema_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, schema_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(schema_embedding)
            )
            similarities.append((db_id, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_schema_metadata(self, database_id: str) -> Optional[SchemaMetadata]:
        """Get schema metadata for a database."""
        return self.schema_metadata.get(database_id)
    
    def save_embeddings(self, filepath: str) -> None:
        """Save embeddings and metadata to file."""
        data = {
            'embeddings': self.schema_embeddings,
            'metadata': {k: asdict(v) for k, v in self.schema_metadata.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> None:
        """Load embeddings and metadata from file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.schema_embeddings = data['embeddings']
            self.schema_metadata = {
                k: SchemaMetadata(**v) for k, v in data['metadata'].items()
            }
            logger.info(f"Loaded embeddings from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def get_embedding_stats(self) -> Dict[str, int]:
        """Get statistics about loaded embeddings."""
        return {
            'total_schemas': len(self.schema_embeddings),
            'databases_by_type': self._count_by_type(),
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count databases by type."""
        type_counts = {}
        for metadata in self.schema_metadata.values():
            db_type = metadata.database_type
            type_counts[db_type] = type_counts.get(db_type, 0) + 1
        return type_counts
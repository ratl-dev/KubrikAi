"""
Mock embedding model for testing without internet access.
"""

import numpy as np
from typing import List


class MockEmbeddingModel:
    """Mock sentence transformer for testing purposes."""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        self.embedding_dim = 384  # Typical dimension for MiniLM
    
    def encode(self, sentences: List[str]) -> np.ndarray:
        """Generate mock embeddings based on text characteristics."""
        embeddings = []
        
        for sentence in sentences:
            # Create deterministic embeddings based on text features
            words = sentence.lower().split()
            
            # Create base embedding with more variation
            seed = hash(sentence) % 2**32
            embedding = np.random.RandomState(seed).normal(0, 0.3, self.embedding_dim)
            
            # Add strong features based on domain keywords
            if any(word in sentence.lower() for word in ['order', 'customer', 'product', 'purchase', 'cart']):
                embedding[0:50] += 1.0  # Strong e-commerce signal
            
            if any(word in sentence.lower() for word in ['metric', 'analytics', 'event', 'conversion', 'funnel']):
                embedding[50:100] += 1.0  # Strong analytics signal
                
            if any(word in sentence.lower() for word in ['employee', 'department', 'salary', 'hire']):
                embedding[100:150] += 1.0  # Strong HR signal
            
            # Add weaker secondary signals
            if any(word in sentence.lower() for word in ['data', 'report', 'analysis']):
                embedding[200:220] += 0.3
                
            if any(word in sentence.lower() for word in ['total', 'count', 'sum', 'top']):
                embedding[220:240] += 0.3
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)
        
        return np.array(embeddings)
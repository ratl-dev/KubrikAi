"""
Simple test script to validate the intelligent routing system works
without requiring external model downloads.
"""

import logging
import numpy as np
from kubrik_ai.routing import (
    SchemaMetadata, DatabaseConfig, UserContext, QueryLimits,
    ValidationLevel, DatabaseRouter, SQLValidator, SecurityEnforcer
)

# Set up logging
logging.basicConfig(level=logging.INFO)


class MockSchemaEmbedder:
    """Mock embedder that doesn't require external model downloads"""
    
    def __init__(self):
        self.schema_embeddings = {}
        self.schema_metadata = {}
    
    def add_database_schema(self, metadata):
        # Create a simple mock embedding based on schema content
        text_content = f"{metadata.database_type} {' '.join(metadata.tables)} {metadata.description or ''}"
        # Simple hash-based embedding for demo
        embedding = np.array([hash(text_content) % 100 / 100.0 for _ in range(5)])
        
        self.schema_embeddings[metadata.database_id] = embedding
        self.schema_metadata[metadata.database_id] = metadata
        print(f"Added mock embedding for {metadata.database_id}")
    
    def find_similar_schemas(self, query, top_k=3):
        if not self.schema_embeddings:
            return []
        
        # Simple mock similarity based on keyword matching
        similarities = []
        query_lower = query.lower()
        
        for db_id, metadata in self.schema_metadata.items():
            score = 0.5  # Base score
            
            # Boost score for keyword matches
            if metadata.database_type.lower() in query_lower:
                score += 0.2
            
            for table in metadata.tables:
                if table.lower() in query_lower:
                    score += 0.1
            
            if metadata.description and any(word in metadata.description.lower() 
                                          for word in query_lower.split()):
                score += 0.1
            
            similarities.append((db_id, min(score, 1.0)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_schema_metadata(self, database_id):
        return self.schema_metadata.get(database_id)
    
    def get_embedding_stats(self):
        return {
            'total_schemas': len(self.schema_embeddings),
            'databases_by_type': {}
        }


def test_core_functionality():
    """Test the core functionality without external dependencies"""
    
    print("🧪 Testing Core Routing Components\n")
    
    # Test SQL Validator
    print("1. Testing SQL Validator...")
    validator = SQLValidator(ValidationLevel.STANDARD)
    
    # Test valid query
    result = validator.validate_query("SELECT * FROM users WHERE id = 1")
    print(f"  Valid query test: {'✅ PASS' if result.is_valid else '❌ FAIL'}")
    
    # Test invalid write query
    result = validator.validate_query("DELETE FROM users WHERE id = 1")
    print(f"  Invalid write query test: {'✅ PASS' if not result.is_valid else '❌ FAIL'}")
    
    # Test Security Enforcer
    print("\n2. Testing Security Enforcer...")
    enforcer = SecurityEnforcer()
    
    try:
        enforcer.enforce_read_only("SELECT * FROM users")
        print("  Read-only enforcement for SELECT: ✅ PASS")
    except Exception as e:
        print(f"  Read-only enforcement for SELECT: ❌ FAIL - {e}")
    
    try:
        enforcer.enforce_read_only("DROP TABLE users")
        print("  Read-only enforcement for DROP: ❌ FAIL - Should have raised exception")
    except Exception:
        print("  Read-only enforcement for DROP: ✅ PASS")
    
    # Test Schema Embedder (Mock)
    print("\n3. Testing Schema Embedder (Mock)...")
    embedder = MockSchemaEmbedder()
    
    # Add test schema
    schema = SchemaMetadata(
        database_id="test_db",
        database_type="postgres",
        tables=["users", "orders"],
        columns={"users": ["id", "name"], "orders": ["id", "user_id", "total"]},
        relationships=[],
        indexes={},
        description="Test e-commerce database",
        domain="ecommerce"
    )
    
    embedder.add_database_schema(schema)
    
    # Test similarity search
    similar = embedder.find_similar_schemas("show user orders")
    print(f"  Schema similarity search: {'✅ PASS' if similar else '❌ FAIL'}")
    if similar:
        print(f"    Found: {similar[0][0]} with score {similar[0][1]:.3f}")
    
    # Test Database Router
    print("\n4. Testing Database Router...")
    router = DatabaseRouter(embedder)
    
    # Add database config
    config = DatabaseConfig(
        database_id="test_db",
        connection_string="postgres://localhost/test",
        specializations=["transactional"]
    )
    router.add_database_config(config)
    
    # Test query classification
    query_type = router.classify_query("SELECT SUM(total) FROM orders GROUP BY user_id")
    print(f"  Query classification: {'✅ PASS' if query_type.value == 'analytical' else '❌ FAIL'}")
    print(f"    Classified as: {query_type.value}")
    
    # Test routing
    try:
        decision = router.route_query("Show me user order data")
        print(f"  Query routing: {'✅ PASS' if decision.selected_database == 'test_db' else '❌ FAIL'}")
        print(f"    Selected: {decision.selected_database}")
        print(f"    Confidence: {decision.confidence_score:.3f}")
        print(f"    Reasoning: {decision.reasoning}")
    except Exception as e:
        print(f"  Query routing: ❌ FAIL - {e}")
    
    print("\n5. Testing System Integration...")
    
    # Test user context
    user_context = UserContext(
        user_id="test_user",
        roles=["analyst"],
        permissions=["table:*"],
        query_limits=QueryLimits(
            max_execution_time_seconds=300,
            max_result_rows=10000,
            max_concurrent_queries=3
        )
    )
    
    # Test security checks
    try:
        enforcer.check_query_limits(user_context)
        print("  User context and limits: ✅ PASS")
    except Exception as e:
        print(f"  User context and limits: ❌ FAIL - {e}")
    
    print("\n🎉 Core functionality tests completed!")
    print("\n" + "="*60)
    
    # Show what would happen in a real routing scenario
    print("\n📊 Simulated Routing Scenario:")
    test_queries = [
        "Show me total sales by product category",
        "Find users who placed orders last month", 
        "Get real-time page view metrics"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        # Classify query
        query_type = router.classify_query(query)
        print(f"Query Type: {query_type.value}")
        
        # Find similar schemas
        similar = embedder.find_similar_schemas(query, top_k=1)
        if similar:
            print(f"Best Match: {similar[0][0]} (similarity: {similar[0][1]:.3f})")
        
        # Validate query (simplified)
        validation_result = validator.validate_query(f"SELECT * FROM table WHERE condition")
        print(f"SQL Valid: {validation_result.is_valid}")
        
        # Security check
        try:
            enforcer.enforce_read_only("SELECT * FROM table")
            print("Security: ✅ PASSED")
        except:
            print("Security: ❌ BLOCKED")


if __name__ == "__main__":
    test_core_functionality()
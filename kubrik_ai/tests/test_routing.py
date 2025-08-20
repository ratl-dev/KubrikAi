"""
Tests for the intelligent routing system
"""

import pytest
import time
from kubrik_ai.routing import (
    SchemaEmbedder, SchemaMetadata, DatabaseRouter, DatabaseConfig,
    SQLValidator, ValidationLevel, SecurityEnforcer, UserContext, QueryLimits,
    IntelligentRoutingSystem, RoutingConfig, QueryRequest, SecurityViolation
)


class TestSchemaEmbedder:
    """Test schema embedding functionality"""
    
    def test_schema_embedding_creation(self):
        """Test creating and managing schema embeddings"""
        embedder = SchemaEmbedder()
        
        # Create test schema
        schema = SchemaMetadata(
            database_id="test_db",
            database_type="postgres",
            tables=["users", "orders"],
            columns={"users": ["id", "name"], "orders": ["id", "user_id", "amount"]},
            relationships=[],
            indexes={},
            description="Test database",
            domain="test"
        )
        
        # Add schema
        embedder.add_database_schema(schema)
        
        # Test that embedding was created
        assert "test_db" in embedder.schema_embeddings
        assert embedder.schema_embeddings["test_db"] is not None
        
        # Test similarity search
        similar = embedder.find_similar_schemas("show user orders", top_k=1)
        assert len(similar) == 1
        assert similar[0][0] == "test_db"
    
    def test_schema_features_extraction(self):
        """Test schema feature extraction"""
        embedder = SchemaEmbedder()
        
        schema = SchemaMetadata(
            database_id="test_db",
            database_type="mysql",
            tables=["products"],
            columns={"products": ["id", "name", "price"]},
            relationships=[],
            indexes={},
            description="Product catalog",
            domain="e-commerce"
        )
        
        features = embedder.extract_schema_features(schema)
        
        assert "mysql" in features.lower()
        assert "products" in features.lower()
        assert "price" in features.lower()
        assert "e-commerce" in features.lower()


class TestDatabaseRouter:
    """Test database routing logic"""
    
    def setup_method(self):
        """Set up test router with sample data"""
        self.embedder = SchemaEmbedder()
        self.router = DatabaseRouter(self.embedder)
        
        # Add test schemas
        schema1 = SchemaMetadata(
            database_id="analytics_db",
            database_type="postgres",
            tables=["sales", "customers"],
            columns={"sales": ["amount", "date"], "customers": ["id", "segment"]},
            relationships=[],
            indexes={},
            description="Analytics database",
            domain="analytics"
        )
        
        schema2 = SchemaMetadata(
            database_id="transactional_db", 
            database_type="mysql",
            tables=["orders", "users"],
            columns={"orders": ["id", "total"], "users": ["id", "email"]},
            relationships=[],
            indexes={},
            description="Transactional database",
            domain="transactions"
        )
        
        self.embedder.add_database_schema(schema1)
        self.embedder.add_database_schema(schema2)
        
        # Add database configs
        config1 = DatabaseConfig(
            database_id="analytics_db",
            connection_string="postgres://localhost/analytics",
            specializations=["analytics"]
        )
        
        config2 = DatabaseConfig(
            database_id="transactional_db",
            connection_string="mysql://localhost/app",
            specializations=["transactional"]
        )
        
        self.router.add_database_config(config1)
        self.router.add_database_config(config2)
    
    def test_query_classification(self):
        """Test query type classification"""
        # Analytical query
        query1 = "SELECT SUM(amount) FROM sales GROUP BY customer_id"
        assert self.router.classify_query(query1).value == "analytical"
        
        # Simple select
        query2 = "SELECT * FROM users WHERE id = 1"
        assert self.router.classify_query(query2).value == "select"
        
        # Time series
        query3 = "SELECT * FROM orders WHERE date > '2023-01-01'"
        assert self.router.classify_query(query3).value == "time_series"
    
    def test_routing_rules(self):
        """Test routing rule application"""
        # Add a routing rule
        self.router.add_routing_rule({
            "name": "analytics_rule",
            "condition": {"query_type": "analytical"},
            "target_database": "analytics_db",
            "priority": 10
        })
        
        # Test routing
        decision = self.router.route_query("SELECT AVG(amount) FROM sales")
        assert decision.selected_database == "analytics_db"
        assert "routing rule" in decision.reasoning.lower()
    
    def test_similarity_based_routing(self):
        """Test similarity-based routing when no rules apply"""
        decision = self.router.route_query("Show me customer sales data")
        
        # Should have routing decision
        assert decision.selected_database in ["analytics_db", "transactional_db"]
        assert decision.confidence_score > 0
        assert len(decision.similarity_scores) > 0


class TestSQLValidator:
    """Test SQL validation functionality"""
    
    def test_basic_validation(self):
        """Test basic SQL validation"""
        validator = SQLValidator(ValidationLevel.BASIC)
        
        # Valid query
        result = validator.validate_query("SELECT * FROM users")
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Invalid syntax
        result = validator.validate_query("SELEC * FROM users")
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_read_only_enforcement(self):
        """Test read-only constraint checking"""
        validator = SQLValidator(ValidationLevel.STANDARD)
        
        # Valid read-only query
        result = validator.validate_query("SELECT * FROM users")
        assert result.is_valid
        
        # Invalid write query
        result = validator.validate_query("DELETE FROM users WHERE id = 1")
        assert not result.is_valid
        assert any("DELETE" in error for error in result.errors)
        
        # Another invalid write query
        result = validator.validate_query("UPDATE users SET name = 'test'")
        assert not result.is_valid
    
    def test_security_checks(self):
        """Test security vulnerability detection"""
        validator = SQLValidator(ValidationLevel.STANDARD)
        
        # Potential SQL injection
        result = validator.validate_query("SELECT * FROM users WHERE id = '1'; DROP TABLE users;--'")
        assert not result.is_valid
        assert any("injection" in error.lower() for error in result.errors)


class TestSecurityEnforcer:
    """Test security enforcement"""
    
    def setup_method(self):
        """Set up test security enforcer"""
        self.enforcer = SecurityEnforcer()
        self.user_context = UserContext(
            user_id="test_user",
            roles=["analyst"],
            permissions=["table:*"],
            query_limits=QueryLimits(
                max_execution_time_seconds=300,
                max_result_rows=10000,
                max_concurrent_queries=2
            )
        )
    
    def test_read_only_enforcement(self):
        """Test read-only query enforcement"""
        # Valid read-only query
        self.enforcer.enforce_read_only("SELECT * FROM users")
        
        # Invalid write queries should raise exception
        with pytest.raises(SecurityViolation):
            self.enforcer.enforce_read_only("DELETE FROM users")
        
        with pytest.raises(SecurityViolation):
            self.enforcer.enforce_read_only("UPDATE users SET name = 'test'")
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # First query should pass
        self.enforcer.check_rate_limits("test_user", queries_per_minute=2)
        
        # Second query should pass
        self.enforcer.check_rate_limits("test_user", queries_per_minute=2)
        
        # Third query should fail
        with pytest.raises(SecurityViolation):
            self.enforcer.check_rate_limits("test_user", queries_per_minute=2)
    
    def test_query_limits(self):
        """Test query limit enforcement"""
        # Normal query should pass
        self.enforcer.check_query_limits(self.user_context)
        
        # Query exceeding row limit should fail
        with pytest.raises(SecurityViolation):
            self.enforcer.check_query_limits(
                self.user_context, 
                estimated_rows=20000
            )
        
        # Query exceeding time limit should fail
        with pytest.raises(SecurityViolation):
            self.enforcer.check_query_limits(
                self.user_context,
                estimated_time=600
            )
    
    def test_query_tracking(self):
        """Test query execution tracking"""
        query_id = "test_query_123"
        
        # Register query start
        self.enforcer.register_query_start(
            query_id, self.user_context, "SELECT * FROM users"
        )
        
        # Check that query is tracked
        active_queries = self.enforcer.get_active_queries()
        assert len(active_queries) == 1
        assert active_queries[0]['query_id'] == query_id
        
        # Register query end
        self.enforcer.register_query_end(query_id)
        
        # Check that query is no longer active
        active_queries = self.enforcer.get_active_queries()
        assert len(active_queries) == 0


class TestIntelligentRoutingSystem:
    """Test the complete intelligent routing system"""
    
    def setup_method(self):
        """Set up test routing system"""
        config = RoutingConfig(
            validation_level=ValidationLevel.STANDARD,
            enable_metrics=False,  # Disable for testing
            max_routing_candidates=2
        )
        
        self.system = IntelligentRoutingSystem(config)
        
        # Add test database
        schema = SchemaMetadata(
            database_id="test_db",
            database_type="postgres",
            tables=["users", "orders"],
            columns={"users": ["id", "name"], "orders": ["id", "user_id", "total"]},
            relationships=[],
            indexes={},
            description="Test database",
            domain="test"
        )
        
        db_config = DatabaseConfig(
            database_id="test_db",
            connection_string="postgres://localhost/test"
        )
        
        self.system.add_database(schema, db_config)
        
        self.user_context = UserContext(
            user_id="test_user",
            roles=["analyst"],
            permissions=["table:*"],
            query_limits=QueryLimits()
        )
    
    def test_successful_routing(self):
        """Test successful query routing and validation"""
        request = QueryRequest(
            query="SELECT * FROM users",
            user_context=self.user_context
        )
        
        response = self.system.route_and_validate_query(request)
        
        assert response.success
        assert response.selected_database == "test_db"
        assert response.routing_decision is not None
        assert response.validation_result is not None
        assert response.validation_result.is_valid
    
    def test_security_violation_handling(self):
        """Test handling of security violations"""
        request = QueryRequest(
            query="DELETE FROM users",
            user_context=self.user_context
        )
        
        response = self.system.route_and_validate_query(request)
        
        assert not response.success
        assert "security violation" in response.error_message.lower()
    
    def test_invalid_query_handling(self):
        """Test handling of invalid SQL queries"""
        request = QueryRequest(
            query="INVALID SQL QUERY",
            user_context=self.user_context
        )
        
        response = self.system.route_and_validate_query(request)
        
        # Should route successfully but fail validation
        assert not response.success
        assert response.selected_database == "test_db"
        assert response.validation_result is not None
        assert not response.validation_result.is_valid
    
    def test_database_recommendations(self):
        """Test database recommendation functionality"""
        recommendations = self.system.get_database_recommendations(
            "show user data", top_k=1
        )
        
        assert len(recommendations) == 1
        assert recommendations[0][0] == "test_db"
        assert recommendations[0][1] > 0  # Similarity score
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.system.get_system_status()
        
        assert 'schema_embeddings' in status
        assert 'routing_config' in status
        assert status['schema_embeddings']['total_schemas'] == 1
        assert status['routing_config']['total_databases'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
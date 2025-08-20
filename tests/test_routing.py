"""
Test suite for KubrikAI database routing functionality.

Tests the 2-stage routing system, policy enforcement, and SQL validation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from kubrikai.core import (
    DatabaseRouter, SchemaEngine, PolicyEngine, SQLValidator,
    DatabaseSchema, TableSchema, DatabasePolicy, QueryContext
)
from kubrikai.core.router import DatabaseInfo


class TestSchemaEngine:
    """Test schema embedding and similarity functionality."""
    
    def test_table_schema_to_text(self):
        """Test table schema text conversion."""
        table = TableSchema(
            name="users",
            columns=[
                {"name": "id", "type": "integer", "nullable": False},
                {"name": "email", "type": "varchar", "nullable": False}
            ],
            primary_keys=["id"],
            foreign_keys=[],
            indexes=[],
            constraints=[]
        )
        
        text = table.to_text()
        assert "Table: users" in text
        assert "Column id: integer" in text
        assert "Primary keys: id" in text
    
    def test_database_schema_registration(self):
        """Test database schema registration and embedding."""
        engine = SchemaEngine()
        
        schema = DatabaseSchema(
            db_id="test_db",
            name="Test Database",
            domain="ecommerce",
            tables=[
                TableSchema(
                    name="users",
                    columns=[{"name": "id", "type": "integer"}],
                    primary_keys=["id"],
                    foreign_keys=[],
                    indexes=[],
                    constraints=[]
                )
            ],
            metadata={}
        )
        
        engine.register_database(schema)
        assert "test_db" in engine.schemas
        assert "test_db" in engine.embeddings
    
    def test_query_intent_analysis(self):
        """Test natural language query intent analysis."""
        engine = SchemaEngine()
        
        query = "Show me the top 10 customers who placed orders last month"
        analysis = engine.analyze_query_intent(query)
        
        assert "ecommerce" in analysis["domain_scores"]
        assert analysis["requirements"]["needs_aggregation"]
        assert analysis["requirements"]["needs_sorting"]
        assert analysis["requirements"]["temporal"]


class TestPolicyEngine:
    """Test policy enforcement and security validation."""
    
    def test_database_policy_creation(self):
        """Test database policy configuration."""
        policy = DatabasePolicy(
            db_id="test_db",
            read_only=True,
            max_rows=500,
            max_execution_time=30
        )
        
        assert policy.db_id == "test_db"
        assert policy.read_only is True
        assert policy.max_rows == 500
    
    def test_sql_validation_read_only(self):
        """Test read-only SQL validation."""
        engine = PolicyEngine()
        
        context = QueryContext(
            user_id="test_user",
            session_id="test_session",
            query="Delete some data",
            db_id="test_db",
            timestamp=datetime.now(),
            user_roles=["analyst"],
            request_metadata={}
        )
        
        violations = engine.validate_sql_query(
            "DELETE FROM users WHERE id = 1",
            "test_db",
            context
        )
        
        assert len(violations) > 0
        assert any(v.policy_type == "read_only_violation" for v in violations)
    
    def test_query_limit_enforcement(self):
        """Test automatic LIMIT clause enforcement."""
        engine = PolicyEngine()
        
        # Register policy with row limit
        policy = DatabasePolicy(db_id="test_db", max_rows=100)
        engine.register_database_policy(policy)
        
        # Test adding LIMIT to query without one
        sql = "SELECT * FROM users"
        limited_sql = engine.enforce_query_limits(sql, "test_db")
        assert "LIMIT 100" in limited_sql
        
        # Test reducing existing LIMIT
        sql = "SELECT * FROM users LIMIT 1000"
        limited_sql = engine.enforce_query_limits(sql, "test_db")
        assert "LIMIT 100" in limited_sql


class TestSQLValidator:
    """Test SQL validation and analysis."""
    
    def test_basic_sql_validation(self):
        """Test basic SQL syntax validation."""
        validator = SQLValidator()
        
        # Valid SQL
        result = validator.validate_sql("SELECT * FROM users LIMIT 10")
        assert result.is_valid is True
        assert "users" in result.tables_accessed
        assert "SELECT" in result.operations
    
    def test_dangerous_pattern_detection(self):
        """Test detection of dangerous SQL patterns."""
        validator = SQLValidator()
        
        # Dangerous DROP statement
        result = validator.validate_sql("DROP TABLE users")
        assert result.is_valid is False
        assert "security violation" in result.error.lower()
    
    def test_query_complexity_analysis(self):
        """Test query complexity estimation."""
        validator = SQLValidator()
        
        complex_sql = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE o.created_at > '2023-01-01'
            GROUP BY u.name
            ORDER BY order_count DESC
            LIMIT 10
        """
        
        result = validator.validate_sql(complex_sql)
        assert result.is_valid is True
        assert result.estimated_complexity > 1
        assert result.has_subqueries is False


class TestDatabaseRouter:
    """Test the main 2-stage database routing system."""
    
    @pytest.fixture
    def router(self):
        """Create a test router with mock data."""
        router = DatabaseRouter()
        
        # Register test databases
        db1_schema = DatabaseSchema(
            db_id="ecommerce_db",
            name="E-commerce Database",
            domain="ecommerce",
            tables=[
                TableSchema(
                    name="orders",
                    columns=[{"name": "id", "type": "integer"}],
                    primary_keys=["id"],
                    foreign_keys=[],
                    indexes=[],
                    constraints=[]
                )
            ],
            metadata={}
        )
        
        db1_info = DatabaseInfo(
            db_id="ecommerce_db",
            name="E-commerce Database",
            domain="ecommerce",
            connection_string="postgresql://localhost/ecommerce",
            last_updated=datetime.now(),
            metadata={},
            performance_metrics={}
        )
        
        db2_schema = DatabaseSchema(
            db_id="analytics_db",
            name="Analytics Database", 
            domain="analytics",
            tables=[
                TableSchema(
                    name="events",
                    columns=[{"name": "id", "type": "integer"}],
                    primary_keys=["id"],
                    foreign_keys=[],
                    indexes=[],
                    constraints=[]
                )
            ],
            metadata={}
        )
        
        db2_info = DatabaseInfo(
            db_id="analytics_db",
            name="Analytics Database",
            domain="analytics", 
            connection_string="postgresql://localhost/analytics",
            last_updated=datetime.now() - timedelta(hours=2),
            metadata={},
            performance_metrics={}
        )
        
        router.register_database(db1_info, db1_schema)
        router.register_database(db2_info, db2_schema)
        
        return router
    
    @pytest.mark.asyncio
    async def test_basic_routing(self, router):
        """Test basic query routing functionality."""
        context = QueryContext(
            user_id="test_user",
            session_id="test_session", 
            query="Show me recent orders",
            db_id="",
            timestamp=datetime.now(),
            user_roles=["analyst"],
            request_metadata={"domain": "ecommerce"}
        )
        
        result = await router.route_query("Show me recent orders", context)
        
        assert result.selected_db_id in ["ecommerce_db", "analytics_db"]
        assert result.confidence_score > 0
        assert len(result.stage1_candidates) > 0
        assert result.routing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_forced_routing(self, router):
        """Test forcing routing to specific database."""
        context = QueryContext(
            user_id="test_user",
            session_id="test_session",
            query="Show me data",
            db_id="ecommerce_db", 
            timestamp=datetime.now(),
            user_roles=["analyst"],
            request_metadata={}
        )
        
        result = await router.route_query(
            "Show me data", 
            context, 
            force_db_id="ecommerce_db"
        )
        
        assert result.selected_db_id == "ecommerce_db"
        assert result.confidence_score == 1.0
    
    def test_routing_stats(self, router):
        """Test routing statistics collection."""
        stats = router.get_routing_stats()
        assert "total_routes" in stats
        assert "performance_metrics" in stats
    
    def test_database_list(self, router):
        """Test database listing functionality."""
        databases = router.get_database_list()
        assert len(databases) == 2
        assert any(db["db_id"] == "ecommerce_db" for db in databases)


if __name__ == "__main__":
    pytest.main([__file__])
"""
Example usage of the Intelligent Routing System
"""

import logging
from kubrik_ai.routing import (
    IntelligentRoutingSystem, RoutingConfig, QueryRequest, 
    SchemaMetadata, DatabaseConfig, UserContext, QueryLimits,
    ValidationLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_sample_databases():
    """Create sample database configurations and schemas."""
    
    # E-commerce database
    ecommerce_schema = SchemaMetadata(
        database_id="ecommerce_db",
        database_type="postgres",
        tables=["users", "products", "orders", "order_items", "categories"],
        columns={
            "users": ["user_id", "email", "created_at", "first_name", "last_name"],
            "products": ["product_id", "name", "price", "category_id", "description"],
            "orders": ["order_id", "user_id", "order_date", "total_amount", "status"],
            "order_items": ["order_item_id", "order_id", "product_id", "quantity", "price"],
            "categories": ["category_id", "name", "parent_category_id"]
        },
        relationships=[
            {"from_table": "orders", "from_column": "user_id", "to_table": "users", "to_column": "user_id"},
            {"from_table": "order_items", "from_column": "order_id", "to_table": "orders", "to_column": "order_id"},
            {"from_table": "order_items", "from_column": "product_id", "to_table": "products", "to_column": "product_id"},
            {"from_table": "products", "from_column": "category_id", "to_table": "categories", "to_column": "category_id"}
        ],
        indexes={
            "users": ["email_idx", "created_at_idx"],
            "products": ["category_idx", "name_idx"],
            "orders": ["user_idx", "date_idx"]
        },
        description="E-commerce platform database with users, products, and orders",
        domain="e-commerce"
    )
    
    ecommerce_config = DatabaseConfig(
        database_id="ecommerce_db",
        connection_string="postgresql://user:pass@localhost:5432/ecommerce",
        max_concurrent_queries=10,
        query_timeout_seconds=300,
        read_only=True,
        max_result_rows=10000,
        cost_weight=1.0,
        performance_weight=1.2,
        specializations=["transactional", "real_time"]
    )
    
    # Analytics warehouse
    analytics_schema = SchemaMetadata(
        database_id="analytics_warehouse",
        database_type="bigquery",
        tables=["fact_sales", "dim_customers", "dim_products", "dim_time"],
        columns={
            "fact_sales": ["sale_id", "customer_key", "product_key", "time_key", "quantity", "revenue"],
            "dim_customers": ["customer_key", "customer_id", "segment", "region", "lifetime_value"],
            "dim_products": ["product_key", "product_id", "category", "brand", "cost"],
            "dim_time": ["time_key", "date", "year", "quarter", "month", "day_of_week"]
        },
        relationships=[
            {"from_table": "fact_sales", "from_column": "customer_key", "to_table": "dim_customers", "to_column": "customer_key"},
            {"from_table": "fact_sales", "from_column": "product_key", "to_table": "dim_products", "to_column": "product_key"},
            {"from_table": "fact_sales", "from_column": "time_key", "to_table": "dim_time", "to_column": "time_key"}
        ],
        indexes={
            "fact_sales": ["customer_idx", "product_idx", "time_idx"],
            "dim_customers": ["segment_idx", "region_idx"]
        },
        description="Data warehouse for business intelligence and analytics",
        domain="analytics"
    )
    
    analytics_config = DatabaseConfig(
        database_id="analytics_warehouse",
        connection_string="bigquery://project/dataset",
        max_concurrent_queries=5,
        query_timeout_seconds=600,
        read_only=True,
        max_result_rows=50000,
        cost_weight=2.0,  # Higher cost
        performance_weight=1.5,
        specializations=["analytics", "aggregation"]
    )
    
    # Real-time metrics database
    metrics_schema = SchemaMetadata(
        database_id="metrics_db",
        database_type="postgres",
        tables=["page_views", "user_sessions", "events"],
        columns={
            "page_views": ["id", "user_id", "page", "timestamp", "duration"],
            "user_sessions": ["session_id", "user_id", "start_time", "end_time", "device"],
            "events": ["event_id", "user_id", "event_type", "timestamp", "properties"]
        },
        relationships=[
            {"from_table": "page_views", "from_column": "user_id", "to_table": "user_sessions", "to_column": "user_id"},
            {"from_table": "events", "from_column": "user_id", "to_table": "user_sessions", "to_column": "user_id"}
        ],
        indexes={
            "page_views": ["user_idx", "timestamp_idx"],
            "user_sessions": ["user_idx", "start_time_idx"],
            "events": ["user_idx", "timestamp_idx", "event_type_idx"]
        },
        description="Real-time user behavior and metrics tracking",
        domain="metrics"
    )
    
    metrics_config = DatabaseConfig(
        database_id="metrics_db",
        connection_string="postgresql://user:pass@localhost:5433/metrics",
        max_concurrent_queries=15,
        query_timeout_seconds=120,
        read_only=True,
        max_result_rows=5000,
        cost_weight=0.8,  # Lower cost
        performance_weight=1.0,
        specializations=["real_time", "time_series"]
    )
    
    return [
        (ecommerce_schema, ecommerce_config),
        (analytics_schema, analytics_config),
        (metrics_schema, metrics_config)
    ]


def main():
    """Demonstrate the intelligent routing system."""
    
    print("🚀 Initializing Intelligent Routing System...")
    
    # Create system with configuration
    config = RoutingConfig(
        validation_level=ValidationLevel.STANDARD,
        enable_metrics=True,
        enable_prometheus=False,  # Disable for demo
        max_routing_candidates=3
    )
    
    routing_system = IntelligentRoutingSystem(config)
    
    # Add sample databases
    print("📊 Adding sample databases...")
    for schema, db_config in create_sample_databases():
        routing_system.add_database(schema, db_config)
    
    # Add routing rules
    print("📋 Adding routing rules...")
    routing_system.add_routing_rule({
        "name": "analytics_queries",
        "condition": {"query_type": "analytical"},
        "target_database": "analytics_warehouse",
        "priority": 10
    })
    
    routing_system.add_routing_rule({
        "name": "realtime_metrics",
        "condition": {"domain": "metrics", "specialization": "real_time"},
        "target_database": "metrics_db",
        "priority": 8
    })
    
    # Create user context
    user_context = UserContext(
        user_id="demo_user",
        roles=["analyst"],
        permissions=["table:*"],
        query_limits=QueryLimits(
            max_execution_time_seconds=300,
            max_result_rows=10000,
            max_concurrent_queries=3
        )
    )
    
    # Test queries
    test_queries = [
        "Show me total revenue by product category for the last quarter",
        "Find the top 10 customers by lifetime value",
        "Get real-time page views for the home page in the last hour",
        "List all orders with their customer information",
        "Calculate average session duration by device type"
    ]
    
    print("\n🔍 Testing query routing...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        
        # Create query request
        request = QueryRequest(
            query=query,
            user_context=user_context,
            metadata={"source": "demo"}
        )
        
        # Route and validate
        response = routing_system.route_and_validate_query(request)
        
        print(f"✅ Success: {response.success}")
        print(f"🎯 Selected Database: {response.selected_database}")
        
        if response.routing_decision:
            print(f"🧠 Query Type: {response.routing_decision.query_classification.value}")
            print(f"🎲 Confidence: {response.routing_decision.confidence_score:.3f}")
            print(f"💭 Reasoning: {response.routing_decision.reasoning}")
            
            print("📊 Similarity Scores:")
            for db_id, score in response.routing_decision.similarity_scores:
                print(f"   {db_id}: {score:.3f}")
        
        if response.validation_result:
            print(f"✔️  Valid SQL: {response.validation_result.is_valid}")
            if response.validation_result.errors:
                print(f"❌ Errors: {', '.join(response.validation_result.errors)}")
            if response.validation_result.warnings:
                print(f"⚠️  Warnings: {', '.join(response.validation_result.warnings)}")
        
        if response.warnings:
            print(f"⚠️  System Warnings: {', '.join(response.warnings)}")
        
        if response.error_message:
            print(f"❌ Error: {response.error_message}")
        
        print(f"⏱️  Processing Time: {response.execution_time_ms:.2f}ms")
        print("\n" + "="*80 + "\n")
    
    # Show system status
    print("📈 System Status:")
    status = routing_system.get_system_status()
    print(f"  Databases: {status['schema_embeddings']['total_schemas']}")
    print(f"  Active Queries: {status['active_queries']}")
    print(f"  Validation Level: {status['validation_level']}")
    
    if 'recent_metrics' in status:
        metrics = status['recent_metrics']
        print(f"  Recent Queries: {metrics['total_queries']}")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
    
    # Show database recommendations for a sample query
    print("\n🎯 Database Recommendations for: 'Show customer analytics trends'")
    recommendations = routing_system.get_database_recommendations(
        "Show customer analytics trends", top_k=3
    )
    
    for db_id, similarity, reasoning in recommendations:
        print(f"  {db_id}: {similarity:.3f} - {reasoning}")
    
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()
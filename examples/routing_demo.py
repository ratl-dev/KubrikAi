"""
Example demonstrating KubrikAI intelligent database routing.

This example shows how to set up multiple databases, register their schemas,
and use the 2-stage routing system to intelligently route queries.
"""

import asyncio
from datetime import datetime, timedelta

from kubrikai.core import (
    DatabaseRouter, DatabaseSchema, TableSchema, 
    DatabasePolicy, QueryContext
)
from kubrikai.core.router import DatabaseInfo


async def main():
    """Main example demonstrating intelligent database routing."""
    
    print("🚀 KubrikAI Intelligent Database Routing Example")
    print("=" * 60)
    
    # Initialize the router
    router = DatabaseRouter()
    
    # Set up example databases
    await setup_example_databases(router)
    
    # Configure policies
    setup_policies(router)
    
    # Run example queries
    await run_example_queries(router)
    
    # Show routing statistics
    show_statistics(router)


async def setup_example_databases(router: DatabaseRouter):
    """Set up example databases with different schemas and domains."""
    
    print("\n📊 Setting up example databases...")
    
    # E-commerce database
    ecommerce_schema = DatabaseSchema(
        db_id="ecommerce_prod",
        name="E-commerce Production",
        domain="ecommerce",
        tables=[
            TableSchema(
                name="customers",
                columns=[
                    {"name": "id", "type": "integer", "nullable": False},
                    {"name": "email", "type": "varchar", "nullable": False},
                    {"name": "name", "type": "varchar", "nullable": False},
                    {"name": "registration_date", "type": "timestamp", "nullable": False}
                ],
                primary_keys=["id"],
                foreign_keys=[],
                indexes=[{"name": "idx_email", "columns": ["email"]}],
                constraints=[]
            ),
            TableSchema(
                name="orders", 
                columns=[
                    {"name": "id", "type": "integer", "nullable": False},
                    {"name": "customer_id", "type": "integer", "nullable": False},
                    {"name": "total_amount", "type": "decimal", "nullable": False},
                    {"name": "order_date", "type": "timestamp", "nullable": False}
                ],
                primary_keys=["id"],
                foreign_keys=[{"column": "customer_id", "references": "customers.id"}],
                indexes=[],
                constraints=[]
            ),
            TableSchema(
                name="products",
                columns=[
                    {"name": "id", "type": "integer", "nullable": False},
                    {"name": "name", "type": "varchar", "nullable": False},
                    {"name": "price", "type": "decimal", "nullable": False},
                    {"name": "category", "type": "varchar", "nullable": True}
                ],
                primary_keys=["id"],
                foreign_keys=[],
                indexes=[{"name": "idx_category", "columns": ["category"]}],
                constraints=[]
            )
        ],
        metadata={"row_counts": {"customers": 50000, "orders": 200000, "products": 5000}}
    )
    
    ecommerce_info = DatabaseInfo(
        db_id="ecommerce_prod",
        name="E-commerce Production",
        domain="ecommerce",
        connection_string="postgresql://localhost/ecommerce_prod",
        last_updated=datetime.now() - timedelta(minutes=30),
        metadata={"environment": "production", "region": "us-east-1"},
        performance_metrics={"avg_query_time": 120.5, "uptime": 99.9}
    )
    
    # Analytics database
    analytics_schema = DatabaseSchema(
        db_id="analytics_warehouse",
        name="Analytics Data Warehouse",
        domain="analytics",
        tables=[
            TableSchema(
                name="user_events",
                columns=[
                    {"name": "id", "type": "bigint", "nullable": False},
                    {"name": "user_id", "type": "integer", "nullable": False},
                    {"name": "event_type", "type": "varchar", "nullable": False},
                    {"name": "event_timestamp", "type": "timestamp", "nullable": False}
                ],
                primary_keys=["id"],
                foreign_keys=[],
                indexes=[{"name": "idx_user_timestamp", "columns": ["user_id", "event_timestamp"]}],
                constraints=[]
            ),
            TableSchema(
                name="daily_metrics",
                columns=[
                    {"name": "date", "type": "date", "nullable": False},
                    {"name": "metric_name", "type": "varchar", "nullable": False},
                    {"name": "metric_value", "type": "decimal", "nullable": False}
                ],
                primary_keys=["date", "metric_name"],
                foreign_keys=[],
                indexes=[],
                constraints=[]
            )
        ],
        metadata={"row_counts": {"user_events": 10000000, "daily_metrics": 30000}}
    )
    
    analytics_info = DatabaseInfo(
        db_id="analytics_warehouse",
        name="Analytics Data Warehouse", 
        domain="analytics",
        connection_string="postgresql://localhost/analytics_warehouse",
        last_updated=datetime.now() - timedelta(hours=1),
        metadata={"environment": "production", "region": "us-west-2"},
        performance_metrics={"avg_query_time": 850.2, "uptime": 99.7}
    )
    
    # HR database
    hr_schema = DatabaseSchema(
        db_id="hr_system",
        name="HR Management System",
        domain="hr",
        tables=[
            TableSchema(
                name="employees",
                columns=[
                    {"name": "id", "type": "integer", "nullable": False},
                    {"name": "name", "type": "varchar", "nullable": False},
                    {"name": "department", "type": "varchar", "nullable": False},
                    {"name": "salary", "type": "decimal", "nullable": True},
                    {"name": "hire_date", "type": "date", "nullable": False}
                ],
                primary_keys=["id"],
                foreign_keys=[],
                indexes=[],
                constraints=[]
            )
        ],
        metadata={"row_counts": {"employees": 2500}}
    )
    
    hr_info = DatabaseInfo(
        db_id="hr_system",
        name="HR Management System",
        domain="hr", 
        connection_string="postgresql://localhost/hr_system",
        last_updated=datetime.now() - timedelta(hours=6),
        metadata={"environment": "production", "region": "us-east-1", "sensitive": True},
        performance_metrics={"avg_query_time": 45.1, "uptime": 99.95}
    )
    
    # Register databases
    router.register_database(ecommerce_info, ecommerce_schema)
    router.register_database(analytics_info, analytics_schema)
    router.register_database(hr_info, hr_schema)
    
    print(f"✅ Registered 3 databases:")
    for db in router.get_database_list():
        print(f"   - {db['name']} ({db['domain']})")


def setup_policies(router: DatabaseRouter):
    """Configure database policies for security and governance."""
    
    print("\n🔐 Configuring database policies...")
    
    # E-commerce policy - moderate restrictions
    ecommerce_policy = DatabasePolicy(
        db_id="ecommerce_prod",
        read_only=True,
        max_rows=1000,
        max_execution_time=60,
        require_where_clause=False,
        freshness_requirement=24  # Data must be < 24 hours old
    )
    
    # Analytics policy - fewer restrictions for data analysis
    analytics_policy = DatabasePolicy(
        db_id="analytics_warehouse", 
        read_only=True,
        max_rows=10000,
        max_execution_time=300,
        require_where_clause=False,
        freshness_requirement=48  # More lenient for analytics
    )
    
    # HR policy - strict security
    hr_policy = DatabasePolicy(
        db_id="hr_system",
        read_only=True,
        max_rows=100,
        max_execution_time=30,
        blocked_columns={"employees": {"salary"}},  # Hide sensitive salary data
        require_where_clause=True,
        domain_restrictions=["hr", "executive"],
        freshness_requirement=12
    )
    
    router.policy_engine.register_database_policy(ecommerce_policy)
    router.policy_engine.register_database_policy(analytics_policy)
    router.policy_engine.register_database_policy(hr_policy)
    
    print("✅ Configured security policies for all databases")


async def run_example_queries(router: DatabaseRouter):
    """Run example queries to demonstrate intelligent routing."""
    
    print("\n🧠 Running example queries with intelligent routing...")
    
    # Test queries with different intents and domains
    test_queries = [
        {
            "query": "Show me the top 10 customers by total order value last month",
            "expected_domain": "ecommerce",
            "user_context": {"domain": "business", "role": "analyst"}
        },
        {
            "query": "What are the daily active user metrics for the past week?",
            "expected_domain": "analytics", 
            "user_context": {"domain": "analytics", "role": "data_scientist"}
        },
        {
            "query": "Find all employees in the engineering department",
            "expected_domain": "hr",
            "user_context": {"domain": "hr", "role": "hr_manager"}
        },
        {
            "query": "Calculate conversion rates by product category",
            "expected_domain": "ecommerce",
            "user_context": {"domain": "marketing", "role": "marketing_analyst"}
        },
        {
            "query": "Show user engagement trends over time",
            "expected_domain": "analytics",
            "user_context": {"domain": "product", "role": "product_manager"}
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Query: {test_case['query']}")
        
        # Create query context
        context = QueryContext(
            user_id=f"user_{i}",
            session_id=f"session_{i}",
            query=test_case['query'],
            db_id="",  # Let the router decide
            timestamp=datetime.now(),
            user_roles=[test_case['user_context']['role']],
            request_metadata=test_case['user_context']
        )
        
        try:
            # Route the query
            result = await router.route_query(test_case['query'], context)
            
            print(f"🎯 Selected Database: {result.selected_db_id}")
            print(f"📊 Confidence Score: {result.confidence_score:.3f}")
            print(f"⚡ Routing Time: {result.routing_time_ms:.1f}ms")
            print(f"💡 Explanation: {result.explanation}")
            
            if result.policy_violations:
                print(f"⚠️  Policy Considerations: {len(result.policy_violations)}")
                for violation in result.policy_violations:
                    print(f"   - {violation.severity}: {violation.message}")
            
            # Show alternative candidates
            if len(result.stage1_candidates) > 1:
                print("🔄 Alternative Candidates:")
                for db_id, similarity in result.stage1_candidates[1:3]:
                    db_name = router.databases[db_id].name
                    print(f"   - {db_name}: {similarity:.3f}")
            
        except Exception as e:
            print(f"❌ Routing failed: {str(e)}")


def show_statistics(router: DatabaseRouter):
    """Display routing and performance statistics."""
    
    print("\n📈 Routing Statistics")
    print("=" * 40)
    
    stats = router.get_routing_stats()
    
    print(f"Total Routes: {stats['total_routes']}")
    if stats['total_routes'] > 0:
        print(f"Average Routing Time: {stats['avg_routing_time_ms']:.1f}ms")
        print(f"Average Confidence: {stats['avg_confidence']:.3f}")
        
        print("\nDatabase Selection Frequency:")
        for db_id, count in stats['database_selections'].items():
            db_name = router.databases[db_id].name
            percentage = (count / stats['total_routes']) * 100
            print(f"  {db_name}: {count} times ({percentage:.1f}%)")
    
    print("\nDatabase Performance Metrics:")
    performance_metrics = stats.get('performance_metrics', {})
    for db_id, metrics in performance_metrics.items():
        db_name = router.databases[db_id].name
        print(f"  {db_name}:")
        print(f"    Avg Query Time: {metrics.get('avg_query_time', 0):.1f}ms")
        print(f"    Success Rate: {metrics.get('success_rate', 0):.1%}")
        print(f"    Total Queries: {metrics.get('total_queries', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
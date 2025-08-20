"""
KubrikAI Command Line Interface

Simple CLI for testing and demonstrating the intelligent database routing system.
"""

import asyncio
import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kubrikai.core import (
    DatabaseRouter, DatabaseSchema, TableSchema,
    DatabasePolicy, QueryContext
)
from kubrikai.core.router import DatabaseInfo


class KubrikAICLI:
    """Command line interface for KubrikAI."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.router = DatabaseRouter()
        self.setup_demo_data()
    
    def setup_demo_data(self):
        """Set up demo databases for testing."""
        # E-commerce demo database
        ecommerce_schema = DatabaseSchema(
            db_id="demo_ecommerce",
            name="Demo E-commerce",
            domain="ecommerce",
            tables=[
                TableSchema(
                    name="customers",
                    columns=[
                        {"name": "id", "type": "integer"},
                        {"name": "email", "type": "varchar"},
                        {"name": "name", "type": "varchar"}
                    ],
                    primary_keys=["id"],
                    foreign_keys=[],
                    indexes=[],
                    constraints=[]
                ),
                TableSchema(
                    name="orders",
                    columns=[
                        {"name": "id", "type": "integer"},
                        {"name": "customer_id", "type": "integer"},
                        {"name": "total", "type": "decimal"}
                    ],
                    primary_keys=["id"],
                    foreign_keys=[{"column": "customer_id", "references": "customers.id"}],
                    indexes=[],
                    constraints=[]
                )
            ],
            metadata={}
        )
        
        ecommerce_info = DatabaseInfo(
            db_id="demo_ecommerce",
            name="Demo E-commerce",
            domain="ecommerce",
            connection_string="postgresql://localhost/demo_ecommerce",
            last_updated=datetime.now(),
            metadata={},
            performance_metrics={}
        )
        
        # Analytics demo database
        analytics_schema = DatabaseSchema(
            db_id="demo_analytics",
            name="Demo Analytics",
            domain="analytics",
            tables=[
                TableSchema(
                    name="events",
                    columns=[
                        {"name": "id", "type": "bigint"},
                        {"name": "user_id", "type": "integer"},
                        {"name": "event_type", "type": "varchar"}
                    ],
                    primary_keys=["id"],
                    foreign_keys=[],
                    indexes=[],
                    constraints=[]
                )
            ],
            metadata={}
        )
        
        analytics_info = DatabaseInfo(
            db_id="demo_analytics",
            name="Demo Analytics",
            domain="analytics",
            connection_string="postgresql://localhost/demo_analytics",
            last_updated=datetime.now(),
            metadata={},
            performance_metrics={}
        )
        
        # Register databases
        self.router.register_database(ecommerce_info, ecommerce_schema)
        self.router.register_database(analytics_info, analytics_schema)
        
        # Set up policies
        ecommerce_policy = DatabasePolicy(
            db_id="demo_ecommerce",
            read_only=True,
            max_rows=1000,
            max_execution_time=30
        )
        
        analytics_policy = DatabasePolicy(
            db_id="demo_analytics",
            read_only=True,
            max_rows=5000,
            max_execution_time=60
        )
        
        self.router.policy_engine.register_database_policy(ecommerce_policy)
        self.router.policy_engine.register_database_policy(analytics_policy)
    
    async def route_query(self, query: str, user_id: str = "cli_user") -> None:
        """Route a query and display results."""
        print(f"\n🧠 Routing Query: {query}")
        print("-" * 60)
        
        context = QueryContext(
            user_id=user_id,
            session_id="cli_session",
            query=query,
            db_id="",
            timestamp=datetime.now(),
            user_roles=["analyst"],
            request_metadata={"source": "cli"}
        )
        
        try:
            result = await self.router.route_query(query, context)
            
            print(f"✅ Selected Database: {result.selected_db_id}")
            print(f"📊 Confidence Score: {result.confidence_score:.3f}")
            print(f"⚡ Routing Time: {result.routing_time_ms:.1f}ms")
            print(f"💡 Explanation: {result.explanation}")
            
            if result.policy_violations:
                print(f"\n⚠️  Policy Violations ({len(result.policy_violations)}):")
                for violation in result.policy_violations:
                    print(f"   - {violation.severity.upper()}: {violation.message}")
            
            if len(result.stage1_candidates) > 1:
                print(f"\n🔄 Stage 1 Candidates:")
                for db_id, similarity in result.stage1_candidates:
                    db_name = self.router.databases[db_id].name
                    print(f"   - {db_name}: {similarity:.3f}")
            
            print(f"\n📈 Stage 2 Scores:")
            for db_id, score in sorted(result.stage2_scores.items(), key=lambda x: x[1], reverse=True):
                db_name = self.router.databases[db_id].name
                print(f"   - {db_name}: {score:.3f}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    def list_databases(self) -> None:
        """List all registered databases."""
        print("\n📊 Registered Databases:")
        print("-" * 40)
        
        databases = self.router.get_database_list()
        for db in databases:
            print(f"• {db['name']} ({db['db_id']})")
            print(f"  Domain: {db['domain']}")
            print(f"  Last Updated: {db['last_updated']}")
            if db['metrics']:
                print(f"  Metrics: {db['metrics']}")
            print()
    
    def show_stats(self) -> None:
        """Show routing statistics."""
        print("\n📈 Routing Statistics:")
        print("-" * 30)
        
        stats = self.router.get_routing_stats()
        print(f"Total Routes: {stats['total_routes']}")
        
        if stats['total_routes'] > 0:
            print(f"Average Routing Time: {stats['avg_routing_time_ms']:.1f}ms")
            print(f"Average Confidence: {stats['avg_confidence']:.3f}")
            
            if stats['database_selections']:
                print("\nDatabase Selection Frequency:")
                for db_id, count in stats['database_selections'].items():
                    db_name = self.router.databases[db_id].name
                    percentage = (count / stats['total_routes']) * 100
                    print(f"  {db_name}: {count} ({percentage:.1f}%)")
    
    async def interactive_mode(self) -> None:
        """Run in interactive mode."""
        print("🚀 KubrikAI Interactive Mode")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self.show_help()
                elif user_input.lower() in ['list', 'ls']:
                    self.list_databases()
                elif user_input.lower() in ['stats', 'statistics']:
                    self.show_stats()
                elif user_input.lower().startswith('route '):
                    query = user_input[6:].strip()
                    if query:
                        await self.route_query(query)
                    else:
                        print("❌ Please provide a query after 'route'")
                elif user_input:
                    # Treat any other input as a query to route
                    await self.route_query(user_input)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def show_help(self) -> None:
        """Show help information."""
        print("\n📚 Available Commands:")
        print("  help, h          - Show this help")
        print("  list, ls         - List registered databases")
        print("  stats            - Show routing statistics")
        print("  route <query>    - Route a specific query")
        print("  quit, exit, q    - Exit the CLI")
        print("\nOr just type any query to route it directly!")
        
        print("\n💡 Example Queries:")
        print("  • Show me recent customer orders")
        print("  • What are the top products by revenue?")
        print("  • Analyze user engagement metrics")
        print("  • Find high-value customers")


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="KubrikAI Intelligent Database Router CLI")
    parser.add_argument("query", nargs="*", help="Query to route")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", action="store_true", help="List registered databases")
    parser.add_argument("--stats", action="store_true", help="Show routing statistics")
    
    args = parser.parse_args()
    
    cli = KubrikAICLI()
    
    if args.interactive:
        await cli.interactive_mode()
    elif args.list:
        cli.list_databases()
    elif args.stats:
        cli.show_stats()
    elif args.query:
        query = " ".join(args.query)
        await cli.route_query(query)
    else:
        # Default to interactive mode if no arguments
        await cli.interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
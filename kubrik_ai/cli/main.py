"""
CLI interface for the Intelligent Routing System
Provides command-line tools for managing databases, routing queries, and system administration.
"""

import click
import json
import yaml
import logging
from typing import Dict, Any
from pathlib import Path

from kubrik_ai.routing import (
    IntelligentRoutingSystem, RoutingConfig, QueryRequest,
    SchemaMetadata, DatabaseConfig, UserContext, QueryLimits,
    ValidationLevel
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """KubrikAI Intelligent Routing System CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config
    ctx.obj['routing_system'] = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    if not config_path:
        return {}
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_routing_system(ctx) -> IntelligentRoutingSystem:
    """Get or create routing system instance"""
    if ctx.obj['routing_system'] is None:
        config_data = load_config(ctx.obj['config_path']) if ctx.obj['config_path'] else {}
        
        # Create routing config
        routing_config = RoutingConfig(
            validation_level=ValidationLevel(config_data.get('routing_config', {}).get('validation_level', 'standard')),
            enable_metrics=config_data.get('routing_config', {}).get('enable_metrics', True),
            enable_prometheus=config_data.get('routing_config', {}).get('enable_prometheus', False),
            max_routing_candidates=config_data.get('routing_config', {}).get('max_routing_candidates', 3)
        )
        
        ctx.obj['routing_system'] = IntelligentRoutingSystem(routing_config)
        
        # Load databases from config if available
        for db_config in config_data.get('databases', []):
            # This would need schema metadata - simplified for demo
            click.echo(f"Database {db_config['id']} configured from file")
    
    return ctx.obj['routing_system']


@cli.command()
@click.argument('query')
@click.option('--user-id', '-u', default='cli_user', help='User identifier')
@click.option('--preferred-db', '-p', help='Preferred database ID')
@click.option('--output', '-o', type=click.Choice(['json', 'table']), default='table', help='Output format')
@click.pass_context
def route(ctx, query, user_id, preferred_db, output):
    """Route a query to the optimal database"""
    routing_system = get_routing_system(ctx)
    
    try:
        # Create user context
        user_context = UserContext(
            user_id=user_id,
            roles=['cli_user'],
            permissions=['table:*'],
            query_limits=QueryLimits()
        )
        
        # Create query request
        request = QueryRequest(
            query=query,
            user_context=user_context,
            preferred_database=preferred_db
        )
        
        # Route the query
        response = routing_system.route_and_validate_query(request)
        
        if output == 'json':
            result = {
                'query_id': response.query_id,
                'success': response.success,
                'selected_database': response.selected_database,
                'confidence_score': response.routing_decision.confidence_score if response.routing_decision else None,
                'reasoning': response.routing_decision.reasoning if response.routing_decision else None,
                'query_type': response.routing_decision.query_classification.value if response.routing_decision else None,
                'execution_time_ms': response.execution_time_ms,
                'error_message': response.error_message
            }
            click.echo(json.dumps(result, indent=2))
        else:
            # Table format
            click.echo(f"\n🎯 Query Routing Result")
            click.echo("=" * 50)
            click.echo(f"Query ID: {response.query_id}")
            click.echo(f"Success: {'✅' if response.success else '❌'}")
            click.echo(f"Selected Database: {response.selected_database}")
            
            if response.routing_decision:
                click.echo(f"Query Type: {response.routing_decision.query_classification.value}")
                click.echo(f"Confidence: {response.routing_decision.confidence_score:.3f}")
                click.echo(f"Reasoning: {response.routing_decision.reasoning}")
                
                if response.routing_decision.similarity_scores:
                    click.echo("\nSimilarity Scores:")
                    for db_id, score in response.routing_decision.similarity_scores:
                        click.echo(f"  {db_id}: {score:.3f}")
            
            if response.error_message:
                click.echo(f"Error: {response.error_message}")
            
            if response.warnings:
                click.echo(f"Warnings: {', '.join(response.warnings)}")
            
            click.echo(f"Processing Time: {response.execution_time_ms:.2f}ms")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj.get('verbose'):
            raise


@cli.command()
@click.argument('database_id')
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def add_database(ctx, database_id, config_file):
    """Add a database to the routing system"""
    routing_system = get_routing_system(ctx)
    
    try:
        # Load database configuration
        with open(config_file, 'r') as f:
            if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                db_data = yaml.safe_load(f)
            else:
                db_data = json.load(f)
        
        # Create schema metadata
        schema = SchemaMetadata(
            database_id=database_id,
            database_type=db_data['type'],
            tables=db_data.get('tables', []),
            columns=db_data.get('columns', {}),
            relationships=db_data.get('relationships', []),
            indexes=db_data.get('indexes', {}),
            description=db_data.get('description'),
            domain=db_data.get('domain')
        )
        
        # Create database config
        config = DatabaseConfig(
            database_id=database_id,
            connection_string=db_data['connection_string'],
            max_concurrent_queries=db_data.get('max_concurrent_queries', 10),
            query_timeout_seconds=db_data.get('query_timeout_seconds', 300),
            read_only=db_data.get('read_only', True),
            max_result_rows=db_data.get('max_result_rows', 10000),
            cost_weight=db_data.get('cost_weight', 1.0),
            performance_weight=db_data.get('performance_weight', 1.0),
            specializations=db_data.get('specializations', [])
        )
        
        # Add to routing system
        routing_system.add_database(schema, config)
        
        click.echo(f"✅ Database '{database_id}' added successfully")
        
    except Exception as e:
        click.echo(f"❌ Error adding database: {e}", err=True)
        if ctx.obj.get('verbose'):
            raise


@cli.command()
@click.pass_context
def list_databases(ctx):
    """List all configured databases"""
    routing_system = get_routing_system(ctx)
    
    try:
        databases = routing_system.database_router.list_available_databases()
        
        if not databases:
            click.echo("No databases configured")
            return
        
        click.echo("\n📊 Configured Databases:")
        click.echo("=" * 30)
        for db_id in databases:
            click.echo(f"  • {db_id}")
        
    except Exception as e:
        click.echo(f"❌ Error listing databases: {e}", err=True)


@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=5, help='Number of recommendations to show')
@click.pass_context
def recommend(ctx, query, top_k):
    """Get database recommendations for a query"""
    routing_system = get_routing_system(ctx)
    
    try:
        recommendations = routing_system.get_database_recommendations(query, top_k)
        
        if not recommendations:
            click.echo("No recommendations available")
            return
        
        click.echo(f"\n🎯 Database Recommendations for: '{query}'")
        click.echo("=" * 60)
        
        for i, (db_id, similarity, reasoning) in enumerate(recommendations, 1):
            click.echo(f"{i}. {db_id}")
            click.echo(f"   Similarity: {similarity:.3f}")
            click.echo(f"   Reasoning: {reasoning}")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Error getting recommendations: {e}", err=True)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and metrics"""
    routing_system = get_routing_system(ctx)
    
    try:
        status = routing_system.get_system_status()
        
        click.echo("\n📈 System Status")
        click.echo("=" * 40)
        click.echo(f"Total Databases: {status['routing_config']['total_databases']}")
        click.echo(f"Total Schemas: {status['schema_embeddings']['total_schemas']}")
        click.echo(f"Active Queries: {status['active_queries']}")
        click.echo(f"Validation Level: {status['validation_level']}")
        click.echo(f"Metrics Enabled: {status['metrics_enabled']}")
        
        if 'recent_metrics' in status and status['recent_metrics']:
            metrics = status['recent_metrics']
            click.echo(f"\nRecent Metrics (last {metrics['time_period_hours']} hours):")
            click.echo(f"  Total Queries: {metrics['total_queries']}")
            click.echo(f"  Successful: {metrics['successful_queries']}")
            click.echo(f"  Failed: {metrics['failed_queries']}")
            click.echo(f"  Success Rate: {metrics['success_rate']:.1%}")
            
            if metrics['database_usage']:
                click.echo(f"\nDatabase Usage:")
                for db_id, count in metrics['database_usage'].items():
                    click.echo(f"  {db_id}: {count} queries")
        
    except Exception as e:
        click.echo(f"❌ Error getting status: {e}", err=True)


@cli.command()
@click.option('--force', is_flag=True, help='Force cleanup without confirmation')
@click.pass_context
def cleanup(ctx, force):
    """Clean up system resources and old data"""
    if not force:
        if not click.confirm('This will clean up old metrics and cached data. Continue?'):
            return
    
    routing_system = get_routing_system(ctx)
    
    try:
        routing_system.cleanup_resources()
        click.echo("✅ System cleanup completed")
        
    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e}", err=True)


@cli.command()
@click.argument('config_file', type=click.Path())
@click.pass_context
def save_config(ctx, config_file):
    """Save current configuration to file"""
    routing_system = get_routing_system(ctx)
    
    try:
        # Save embeddings and configuration
        routing_system.save_configuration(config_file)
        click.echo(f"✅ Configuration saved to {config_file}")
        
    except Exception as e:
        click.echo(f"❌ Error saving configuration: {e}", err=True)


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.pass_context
def load_config_cmd(ctx, config_file):
    """Load configuration from file"""
    routing_system = get_routing_system(ctx)
    
    try:
        # Load embeddings and configuration
        routing_system.load_configuration(config_file)
        click.echo(f"✅ Configuration loaded from {config_file}")
        
    except Exception as e:
        click.echo(f"❌ Error loading configuration: {e}", err=True)


if __name__ == '__main__':
    cli()
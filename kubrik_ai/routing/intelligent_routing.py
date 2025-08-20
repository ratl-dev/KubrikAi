"""
Intelligent Routing System - Main orchestrator that combines all components
to provide intelligent database routing for SQL queries.
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .schema_embedder import SchemaEmbedder, SchemaMetadata
from .database_router import DatabaseRouter, DatabaseConfig, RoutingDecision
from .sql_validator import SQLValidator, ValidationResult, ValidationLevel
from .security_enforcer import SecurityEnforcer, UserContext, QueryLimits, SecurityViolation
from .metrics_logger import MetricsLogger

logger = logging.getLogger(__name__)


@dataclass
class RoutingConfig:
    """Configuration for the intelligent routing system"""
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    enable_metrics: bool = True
    enable_prometheus: bool = True
    max_routing_candidates: int = 3
    default_query_timeout: int = 300
    default_max_rows: int = 10000


@dataclass
class QueryRequest:
    """A query request to be routed and executed"""
    query: str
    user_context: UserContext
    preferred_database: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryResponse:
    """Response from query routing and execution"""
    query_id: str
    success: bool
    selected_database: str
    routing_decision: RoutingDecision
    validation_result: ValidationResult
    execution_time_ms: float
    results: Optional[Any] = None
    error_message: Optional[str] = None
    warnings: List[str] = None


class IntelligentRoutingSystem:
    """
    Main orchestrator for intelligent SQL database routing.
    
    Combines schema embeddings, routing logic, validation, security enforcement,
    and metrics collection into a unified system.
    """
    
    def __init__(self, config: RoutingConfig = None):
        """
        Initialize the intelligent routing system.
        
        Args:
            config: System configuration
        """
        self.config = config or RoutingConfig()
        
        # Initialize core components
        self.schema_embedder = SchemaEmbedder()
        self.database_router = DatabaseRouter(self.schema_embedder)
        self.sql_validator = SQLValidator(self.config.validation_level)
        self.security_enforcer = SecurityEnforcer()
        
        # Initialize metrics if enabled
        if self.config.enable_metrics:
            self.metrics_logger = MetricsLogger(self.config.enable_prometheus)
        else:
            self.metrics_logger = None
        
        logger.info("Intelligent routing system initialized")
    
    def add_database(self, schema_metadata: SchemaMetadata, 
                    database_config: DatabaseConfig) -> None:
        """
        Add a database to the routing system.
        
        Args:
            schema_metadata: Schema metadata for embeddings
            database_config: Database configuration for routing
        """
        # Add to schema embedder
        self.schema_embedder.add_database_schema(schema_metadata)
        
        # Add to router
        self.database_router.add_database_config(database_config)
        
        logger.info(f"Added database: {schema_metadata.database_id}")
    
    def add_routing_rule(self, rule: Dict[str, Any]) -> None:
        """Add a routing rule to the system."""
        self.database_router.add_routing_rule(rule)
    
    def route_and_validate_query(self, request: QueryRequest) -> QueryResponse:
        """
        Route a query to the appropriate database and validate it.
        
        Args:
            request: Query request to process
            
        Returns:
            Query response with routing decision and validation results
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        warnings = []
        
        logger.info(f"Processing query {query_id} for user {request.user_context.user_id}")
        
        try:
            # Step 1: Security checks
            self._perform_security_checks(request, query_id)
            
            # Step 2: Route the query
            routing_start = time.time()
            routing_decision = self._route_query(request)
            routing_time = (time.time() - routing_start) * 1000
            
            # Step 3: Validate the query
            validation_start = time.time()
            validation_result = self._validate_query(request, routing_decision.selected_database)
            validation_time = (time.time() - validation_start) * 1000
            
            # Step 4: Log metrics
            if self.metrics_logger:
                self.metrics_logger.log_query_routed(
                    user_id=request.user_context.user_id,
                    database_id=routing_decision.selected_database,
                    query_type=routing_decision.query_classification.value,
                    routing_time_ms=routing_time,
                    confidence_score=routing_decision.confidence_score,
                    similarity_scores=routing_decision.similarity_scores
                )
                
                self.metrics_logger.log_validation_completed(
                    user_id=request.user_context.user_id,
                    database_id=routing_decision.selected_database,
                    validation_time_ms=validation_time,
                    is_valid=validation_result.is_valid,
                    validation_level=self.config.validation_level.value,
                    errors=validation_result.errors
                )
            
            # Compile warnings
            warnings.extend(validation_result.warnings)
            if routing_decision.confidence_score < 0.7:
                warnings.append(f"Low routing confidence: {routing_decision.confidence_score:.2f}")
            
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                query_id=query_id,
                success=validation_result.is_valid,
                selected_database=routing_decision.selected_database,
                routing_decision=routing_decision,
                validation_result=validation_result,
                execution_time_ms=total_time,
                warnings=warnings,
                error_message='; '.join(validation_result.errors) if validation_result.errors else None
            )
            
        except SecurityViolation as e:
            # Security violation
            error_msg = f"Security violation: {str(e)}"
            logger.warning(f"Query {query_id} blocked: {error_msg}")
            
            if self.metrics_logger:
                self.metrics_logger.log_security_check(
                    user_id=request.user_context.user_id,
                    check_type="query_security",
                    success=False,
                    error_message=str(e)
                )
            
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                query_id=query_id,
                success=False,
                selected_database="",
                routing_decision=None,
                validation_result=None,
                execution_time_ms=total_time,
                error_message=error_msg
            )
            
        except Exception as e:
            # Unexpected error
            error_msg = f"Routing failed: {str(e)}"
            logger.error(f"Query {query_id} failed: {error_msg}", exc_info=True)
            
            if self.metrics_logger:
                self.metrics_logger.log_error(
                    user_id=request.user_context.user_id,
                    error_type="routing_error",
                    error_message=str(e)
                )
            
            total_time = (time.time() - start_time) * 1000
            
            return QueryResponse(
                query_id=query_id,
                success=False,
                selected_database="",
                routing_decision=None,
                validation_result=None,
                execution_time_ms=total_time,
                error_message=error_msg
            )
        
        finally:
            # Clean up any registered query
            self.security_enforcer.register_query_end(query_id)
    
    def _perform_security_checks(self, request: QueryRequest, query_id: str) -> None:
        """Perform security checks on the query request."""
        # Register query start
        self.security_enforcer.register_query_start(
            query_id, request.user_context, request.query
        )
        
        # Check read-only constraint
        self.security_enforcer.enforce_read_only(request.query)
        
        # Check rate limits
        self.security_enforcer.check_rate_limits(request.user_context.user_id)
        
        # Check query limits (basic checks without execution plan)
        self.security_enforcer.check_query_limits(request.user_context)
        
        if self.metrics_logger:
            self.metrics_logger.log_security_check(
                user_id=request.user_context.user_id,
                check_type="all_security_checks",
                success=True
            )
    
    def _route_query(self, request: QueryRequest) -> RoutingDecision:
        """Route the query to the appropriate database."""
        # Use preferred database if specified and available
        if request.preferred_database:
            available_dbs = self.database_router.list_available_databases()
            if request.preferred_database in available_dbs:
                # Still classify the query for metrics
                query_type = self.database_router.classify_query(request.query)
                
                return RoutingDecision(
                    selected_database=request.preferred_database,
                    confidence_score=1.0,
                    reasoning="User-specified preferred database",
                    similarity_scores=[(request.preferred_database, 1.0)],
                    query_classification=query_type,
                    fallback_databases=[]
                )
        
        # Use intelligent routing
        return self.database_router.route_query(
            request.query, 
            top_k_candidates=self.config.max_routing_candidates
        )
    
    def _validate_query(self, request: QueryRequest, database_id: str) -> ValidationResult:
        """Validate the query for the selected database."""
        # Get database configuration for dialect
        db_config = self.database_router.get_database_config(database_id)
        schema_metadata = self.schema_embedder.get_schema_metadata(database_id)
        
        # Determine SQL dialect
        dialect = "postgres"  # Default
        if schema_metadata:
            if schema_metadata.database_type == "mysql":
                dialect = "mysql"
            elif schema_metadata.database_type == "bigquery":
                dialect = "bigquery"
            elif schema_metadata.database_type == "snowflake":
                dialect = "snowflake"
        
        # Validate the query
        return self.sql_validator.validate_query(
            query=request.query,
            dialect=dialect,
            connection=None  # No connection for now, using static validation
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the routing system."""
        schema_stats = self.schema_embedder.get_embedding_stats()
        routing_stats = self.database_router.get_routing_stats()
        
        status = {
            'schema_embeddings': schema_stats,
            'routing_config': routing_stats,
            'active_queries': len(self.security_enforcer.get_active_queries()),
            'validation_level': self.config.validation_level.value,
            'metrics_enabled': self.config.enable_metrics
        }
        
        if self.metrics_logger:
            metrics_summary = self.metrics_logger.get_metrics_summary(hours=1)
            status['recent_metrics'] = metrics_summary
        
        return status
    
    def get_database_recommendations(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Get database recommendations for a query without full routing.
        
        Args:
            query: Query to analyze
            top_k: Number of recommendations to return
            
        Returns:
            List of (database_id, similarity_score, reasoning) tuples
        """
        # Get similarity scores
        similarity_scores = self.schema_embedder.find_similar_schemas(query, top_k)
        
        recommendations = []
        for db_id, similarity in similarity_scores:
            schema_metadata = self.schema_embedder.get_schema_metadata(db_id)
            
            reasoning = f"Schema similarity: {similarity:.3f}"
            if schema_metadata:
                reasoning += f", Type: {schema_metadata.database_type}"
                if schema_metadata.domain:
                    reasoning += f", Domain: {schema_metadata.domain}"
            
            recommendations.append((db_id, similarity, reasoning))
        
        return recommendations
    
    def save_configuration(self, filepath: str) -> None:
        """Save embeddings and configuration to file."""
        self.schema_embedder.save_embeddings(filepath)
        logger.info(f"Configuration saved to {filepath}")
    
    def load_configuration(self, filepath: str) -> None:
        """Load embeddings and configuration from file."""
        self.schema_embedder.load_embeddings(filepath)
        logger.info(f"Configuration loaded from {filepath}")
    
    def cleanup_resources(self) -> None:
        """Clean up system resources and old data."""
        if self.metrics_logger:
            self.metrics_logger.cleanup_old_events()
        
        self.security_enforcer.cleanup_expired_data()
        
        logger.info("System resources cleaned up")
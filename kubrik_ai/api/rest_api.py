"""
REST API for the Intelligent Routing System
Provides endpoints for query routing, database management, and system monitoring.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime

from kubrik_ai.routing import (
    IntelligentRoutingSystem, RoutingConfig, QueryRequest, QueryResponse,
    SchemaMetadata, DatabaseConfig, UserContext, QueryLimits,
    ValidationLevel, SecurityViolation
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KubrikAI Intelligent Routing API",
    description="API for intelligent SQL database routing with schema embeddings and security enforcement",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# Global routing system instance
routing_system: Optional[IntelligentRoutingSystem] = None


# Pydantic models for API
class QueryRequestModel(BaseModel):
    query: str = Field(..., description="SQL query or natural language query to route")
    user_id: str = Field(..., description="User identifier")
    roles: List[str] = Field(default=["user"], description="User roles")
    permissions: List[str] = Field(default=["table:*"], description="User permissions")
    preferred_database: Optional[str] = Field(None, description="Preferred database ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional query metadata")


class QueryResponseModel(BaseModel):
    query_id: str
    success: bool
    selected_database: str
    confidence_score: Optional[float] = None
    reasoning: Optional[str] = None
    query_type: Optional[str] = None
    similarity_scores: Optional[List[Dict[str, float]]] = None
    validation_errors: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None
    execution_time_ms: float
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None


class DatabaseSchemaModel(BaseModel):
    database_id: str
    database_type: str
    tables: List[str]
    columns: Dict[str, List[str]]
    relationships: List[Dict[str, str]] = Field(default_factory=list)
    indexes: Dict[str, List[str]] = Field(default_factory=dict)
    description: Optional[str] = None
    domain: Optional[str] = None


class DatabaseConfigModel(BaseModel):
    database_id: str
    connection_string: str
    max_concurrent_queries: int = 10
    query_timeout_seconds: int = 300
    read_only: bool = True
    max_result_rows: int = 10000
    cost_weight: float = 1.0
    performance_weight: float = 1.0
    specializations: List[str] = Field(default_factory=list)


class RoutingRuleModel(BaseModel):
    name: str
    condition: Dict[str, Any]
    target_database: str
    priority: int = 1


class SystemStatusModel(BaseModel):
    total_databases: int
    total_schemas: int
    active_queries: int
    validation_level: str
    metrics_enabled: bool
    uptime_seconds: float
    recent_metrics: Optional[Dict[str, Any]] = None


class DatabaseRecommendationModel(BaseModel):
    database_id: str
    similarity_score: float
    reasoning: str


# Dependency to get authenticated user (simplified for demo)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Extract user from token (simplified implementation)"""
    # In a real implementation, this would validate the JWT token
    # For demo purposes, we'll extract user_id from the token
    try:
        # Simple token format: "Bearer user_id"
        return credentials.credentials
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


# Initialize the routing system
@app.on_event("startup")
async def startup_event():
    """Initialize the routing system on startup"""
    global routing_system
    
    logger.info("Initializing Intelligent Routing System...")
    
    # Configure the system
    config = RoutingConfig(
        validation_level=ValidationLevel.STANDARD,
        enable_metrics=True,
        enable_prometheus=True,
        max_routing_candidates=3
    )
    
    routing_system = IntelligentRoutingSystem(config)
    logger.info("Intelligent Routing System initialized successfully")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "kubrik-ai-routing"
    }


# Main routing endpoint
@app.post("/query/route", response_model=QueryResponseModel)
async def route_query(
    request: QueryRequestModel,
    current_user: str = Depends(get_current_user)
) -> QueryResponseModel:
    """
    Route a query to the optimal database and validate it.
    """
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        # Create user context
        user_context = UserContext(
            user_id=request.user_id,
            roles=request.roles,
            permissions=request.permissions,
            query_limits=QueryLimits()  # Use defaults
        )
        
        # Create query request
        query_request = QueryRequest(
            query=request.query,
            user_context=user_context,
            preferred_database=request.preferred_database,
            metadata=request.metadata
        )
        
        # Route the query
        response = routing_system.route_and_validate_query(query_request)
        
        # Convert to API response model
        similarity_scores = []
        if response.routing_decision:
            similarity_scores = [
                {"database_id": db_id, "score": score}
                for db_id, score in response.routing_decision.similarity_scores
            ]
        
        return QueryResponseModel(
            query_id=response.query_id,
            success=response.success,
            selected_database=response.selected_database,
            confidence_score=response.routing_decision.confidence_score if response.routing_decision else None,
            reasoning=response.routing_decision.reasoning if response.routing_decision else None,
            query_type=response.routing_decision.query_classification.value if response.routing_decision else None,
            similarity_scores=similarity_scores,
            validation_errors=response.validation_result.errors if response.validation_result else None,
            validation_warnings=response.validation_result.warnings if response.validation_result else None,
            execution_time_ms=response.execution_time_ms,
            error_message=response.error_message,
            warnings=response.warnings
        )
        
    except SecurityViolation as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Security violation: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Query routing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query routing failed: {str(e)}"
        )


# Database management endpoints
@app.post("/database/add")
async def add_database(
    schema: DatabaseSchemaModel,
    config: DatabaseConfigModel,
    current_user: str = Depends(get_current_user)
):
    """Add a new database to the routing system"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        # Convert to internal models
        schema_metadata = SchemaMetadata(
            database_id=schema.database_id,
            database_type=schema.database_type,
            tables=schema.tables,
            columns=schema.columns,
            relationships=schema.relationships,
            indexes=schema.indexes,
            description=schema.description,
            domain=schema.domain
        )
        
        database_config = DatabaseConfig(
            database_id=config.database_id,
            connection_string=config.connection_string,
            max_concurrent_queries=config.max_concurrent_queries,
            query_timeout_seconds=config.query_timeout_seconds,
            read_only=config.read_only,
            max_result_rows=config.max_result_rows,
            cost_weight=config.cost_weight,
            performance_weight=config.performance_weight,
            specializations=config.specializations
        )
        
        # Add to routing system
        routing_system.add_database(schema_metadata, database_config)
        
        return {
            "success": True,
            "message": f"Database {schema.database_id} added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add database: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add database: {str(e)}"
        )


@app.post("/routing/rule/add")
async def add_routing_rule(
    rule: RoutingRuleModel,
    current_user: str = Depends(get_current_user)
):
    """Add a routing rule to the system"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        routing_system.add_routing_rule({
            "name": rule.name,
            "condition": rule.condition,
            "target_database": rule.target_database,
            "priority": rule.priority
        })
        
        return {
            "success": True,
            "message": f"Routing rule '{rule.name}' added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add routing rule: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add routing rule: {str(e)}"
        )


@app.get("/database/recommendations")
async def get_database_recommendations(
    query: str,
    top_k: int = 5,
    current_user: str = Depends(get_current_user)
) -> List[DatabaseRecommendationModel]:
    """Get database recommendations for a query"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        recommendations = routing_system.get_database_recommendations(query, top_k)
        
        return [
            DatabaseRecommendationModel(
                database_id=db_id,
                similarity_score=score,
                reasoning=reasoning
            )
            for db_id, score, reasoning in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get recommendations: {str(e)}"
        )


@app.get("/system/status", response_model=SystemStatusModel)
async def get_system_status(
    current_user: str = Depends(get_current_user)
) -> SystemStatusModel:
    """Get system status and metrics"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        status = routing_system.get_system_status()
        
        return SystemStatusModel(
            total_databases=status['routing_config']['total_databases'],
            total_schemas=status['schema_embeddings']['total_schemas'],
            active_queries=status['active_queries'],
            validation_level=status['validation_level'],
            metrics_enabled=status['metrics_enabled'],
            uptime_seconds=0.0,  # Would track actual uptime in production
            recent_metrics=status.get('recent_metrics')
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


@app.get("/database/list")
async def list_databases(
    current_user: str = Depends(get_current_user)
) -> List[str]:
    """List all available databases"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    try:
        return routing_system.database_router.list_available_databases()
    except Exception as e:
        logger.error(f"Failed to list databases: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list databases: {str(e)}"
        )


@app.post("/system/cleanup")
async def cleanup_system(
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Clean up system resources (background task)"""
    if not routing_system:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Routing system not initialized"
        )
    
    background_tasks.add_task(routing_system.cleanup_resources)
    
    return {
        "success": True,
        "message": "System cleanup started in background"
    }


# Error handlers
@app.exception_handler(SecurityViolation)
async def security_violation_handler(request, exc):
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Security violation: {str(exc)}"
    )


if __name__ == "__main__":
    uvicorn.run(
        "rest_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
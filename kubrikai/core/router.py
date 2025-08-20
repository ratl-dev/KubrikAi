"""
Intelligent database router for KubrikAI.

Implements 2-stage routing: schema embedding similarity + rule-based classification
to select the optimal database for query execution.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from .schema_engine import SchemaEngine, DatabaseSchema
from .policy_engine import PolicyEngine, QueryContext, PolicyViolation, PolicyAction
from .sql_validator import SQLValidator

logger = logging.getLogger(__name__)


@dataclass
class DatabaseInfo:
    """Information about a registered database."""
    db_id: str
    name: str
    domain: str
    connection_string: str
    last_updated: datetime
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]


@dataclass
class RoutingResult:
    """Result of database routing process."""
    selected_db_id: str
    confidence_score: float
    stage1_candidates: List[Tuple[str, float]]  # (db_id, similarity_score)
    stage2_scores: Dict[str, float]  # db_id -> final_score
    policy_violations: List[PolicyViolation]
    routing_time_ms: float
    explanation: str


@dataclass
class QueryExecutionResult:
    """Result of query execution."""
    success: bool
    sql: str
    rows_returned: int
    execution_time_ms: float
    data: Optional[List[Dict[str, Any]]]
    error: Optional[str]
    warnings: List[str]


class DatabaseRouter:
    """
    Core intelligent database router.
    
    Implements 2-stage routing system:
    1. Schema embedding similarity for shortlisting databases
    2. Rule-based classification considering domain, policy, and freshness
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the database router."""
        self.schema_engine = SchemaEngine(embedding_model)
        self.policy_engine = PolicyEngine()
        self.sql_validator = SQLValidator()
        
        self.databases: Dict[str, DatabaseInfo] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Routing configuration
        self.stage1_candidates = 5  # Number of candidates from similarity
        self.similarity_threshold = 0.1  # Lower threshold for mock embeddings
        self.confidence_threshold = 0.3
        
        # Scoring weights for stage 2
        self.weights = {
            "similarity": 0.4,
            "domain_match": 0.3,
            "freshness": 0.15,
            "performance": 0.1,
            "policy_compliance": 0.05
        }
        
        logger.info("Initialized DatabaseRouter with 2-stage routing")
    
    def register_database(
        self, 
        db_info: DatabaseInfo, 
        schema: DatabaseSchema
    ) -> None:
        """Register a database with the router."""
        self.databases[db_info.db_id] = db_info
        self.schema_engine.register_database(schema)
        
        # Initialize performance metrics
        self.performance_metrics[db_info.db_id] = {
            "avg_query_time": 0.0,
            "success_rate": 1.0,
            "total_queries": 0,
            "last_response_time": 0.0
        }
        
        logger.info(f"Registered database {db_info.db_id} ({db_info.name})")
    
    async def route_query(
        self, 
        query: str, 
        context: QueryContext,
        force_db_id: Optional[str] = None
    ) -> RoutingResult:
        """
        Route a natural language query to the best database.
        
        Args:
            query: Natural language query
            context: Query execution context
            force_db_id: Force routing to specific database (for testing)
        
        Returns:
            Routing result with selected database and metadata
        """
        start_time = time.time()
        
        try:
            # If forced to specific database, skip routing
            if force_db_id:
                if force_db_id not in self.databases:
                    raise ValueError(f"Database {force_db_id} not found")
                
                # Still check policies
                violations = self.policy_engine.evaluate_database_access(force_db_id, context)
                
                return RoutingResult(
                    selected_db_id=force_db_id,
                    confidence_score=1.0,
                    stage1_candidates=[(force_db_id, 1.0)],
                    stage2_scores={force_db_id: 1.0},
                    policy_violations=violations,
                    routing_time_ms=(time.time() - start_time) * 1000,
                    explanation=f"Forced routing to {force_db_id}"
                )
            
            # Stage 1: Schema embedding similarity
            stage1_candidates = await self._stage1_similarity_routing(query)
            
            if not stage1_candidates:
                raise ValueError("No databases found matching similarity threshold")
            
            # Stage 2: Rule-based classification
            stage2_scores = await self._stage2_classification(
                query, stage1_candidates, context
            )
            
            # Select best database
            best_db_id = max(stage2_scores.items(), key=lambda x: x[1])[0]
            confidence = stage2_scores[best_db_id]
            
            # Check policies for selected database
            violations = self.policy_engine.evaluate_database_access(best_db_id, context)
            
            # If selected database has critical violations, try alternatives
            critical_violations = [v for v in violations if v.severity == "critical"]
            if critical_violations:
                for db_id, score in sorted(stage2_scores.items(), key=lambda x: x[1], reverse=True)[1:]:
                    alt_violations = self.policy_engine.evaluate_database_access(db_id, context)
                    alt_critical = [v for v in alt_violations if v.severity == "critical"]
                    if not alt_critical:
                        best_db_id = db_id
                        confidence = score
                        violations = alt_violations
                        break
            
            routing_time = (time.time() - start_time) * 1000
            
            # Generate explanation
            explanation = self._generate_routing_explanation(
                best_db_id, stage1_candidates, stage2_scores, violations
            )
            
            result = RoutingResult(
                selected_db_id=best_db_id,
                confidence_score=confidence,
                stage1_candidates=stage1_candidates,
                stage2_scores=stage2_scores,
                policy_violations=violations,
                routing_time_ms=routing_time,
                explanation=explanation
            )
            
            # Log routing decision
            self._log_routing_decision(query, context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query routing: {str(e)}")
            raise
    
    async def _stage1_similarity_routing(self, query: str) -> List[Tuple[str, float]]:
        """Stage 1: Find databases with similar schemas to query requirements."""
        # Generate query embedding
        query_embedding = self.schema_engine.embed_query(query)
        
        # Get similar databases
        candidates = self.schema_engine.get_similar_databases(
            query_embedding,
            top_k=self.stage1_candidates,
            threshold=self.similarity_threshold
        )
        
        logger.info(f"Stage 1: Found {len(candidates)} candidate databases")
        return candidates
    
    async def _stage2_classification(
        self, 
        query: str,
        stage1_candidates: List[Tuple[str, float]],
        context: QueryContext
    ) -> Dict[str, float]:
        """Stage 2: Apply rules and classification to rank candidates."""
        query_analysis = self.schema_engine.analyze_query_intent(query)
        scores = {}
        
        for db_id, similarity_score in stage1_candidates:
            db_info = self.databases[db_id]
            
            # Calculate individual scoring components
            domain_score = self._calculate_domain_score(db_info, query_analysis)
            freshness_score = self._calculate_freshness_score(db_info, context.timestamp)
            performance_score = self._calculate_performance_score(db_id)
            policy_score = self._calculate_policy_score(db_id, context)
            schema_compatibility = self.schema_engine.get_schema_compatibility(db_id, query_analysis)
            
            # Combined weighted score
            final_score = (
                self.weights["similarity"] * similarity_score +
                self.weights["domain_match"] * domain_score +
                self.weights["freshness"] * freshness_score +
                self.weights["performance"] * performance_score +
                self.weights["policy_compliance"] * policy_score
            )
            
            # Boost score with schema compatibility
            final_score = final_score * (0.5 + 0.5 * schema_compatibility)
            
            scores[db_id] = final_score
            
            logger.debug(f"Database {db_id} scores: similarity={similarity_score:.3f}, "
                        f"domain={domain_score:.3f}, freshness={freshness_score:.3f}, "
                        f"performance={performance_score:.3f}, policy={policy_score:.3f}, "
                        f"compatibility={schema_compatibility:.3f}, final={final_score:.3f}")
        
        return scores
    
    def _calculate_domain_score(self, db_info: DatabaseInfo, query_analysis: Dict[str, Any]) -> float:
        """Calculate domain matching score."""
        domain_scores = query_analysis.get("domain_scores", {})
        if not domain_scores:
            return 0.5  # Neutral if no domain detected
        
        if db_info.domain in domain_scores:
            max_score = max(domain_scores.values())
            return domain_scores[db_info.domain] / max_score
        
        return 0.1  # Low score for domain mismatch
    
    def _calculate_freshness_score(self, db_info: DatabaseInfo, query_time: datetime) -> float:
        """Calculate data freshness score."""
        hours_old = (query_time - db_info.last_updated).total_seconds() / 3600
        
        # Score decreases with age
        if hours_old <= 1:
            return 1.0
        elif hours_old <= 24:
            return 0.8
        elif hours_old <= 168:  # 1 week
            return 0.6
        elif hours_old <= 720:  # 1 month
            return 0.4
        else:
            return 0.2
    
    def _calculate_performance_score(self, db_id: str) -> float:
        """Calculate performance score based on historical metrics."""
        metrics = self.performance_metrics.get(db_id, {})
        
        success_rate = metrics.get("success_rate", 1.0)
        avg_time = metrics.get("avg_query_time", 0.0)
        
        # Normalize response time (assume 1 second is baseline)
        time_score = max(0.1, 1.0 - (avg_time / 1000.0))
        
        return (success_rate * 0.7) + (time_score * 0.3)
    
    def _calculate_policy_score(self, db_id: str, context: QueryContext) -> float:
        """Calculate policy compliance score."""
        violations = self.policy_engine.evaluate_database_access(db_id, context)
        
        if not violations:
            return 1.0
        
        # Penalty based on violation severity
        penalty = 0.0
        for violation in violations:
            if violation.severity == "critical":
                penalty += 0.8
            elif violation.severity == "high":
                penalty += 0.4
            elif violation.severity == "medium":
                penalty += 0.2
            else:
                penalty += 0.1
        
        return max(0.0, 1.0 - penalty)
    
    def _generate_routing_explanation(
        self, 
        selected_db_id: str,
        stage1_candidates: List[Tuple[str, float]],
        stage2_scores: Dict[str, float],
        violations: List[PolicyViolation]
    ) -> str:
        """Generate human-readable explanation for routing decision."""
        db_info = self.databases[selected_db_id]
        
        explanation_parts = [
            f"Selected database: {db_info.name} ({selected_db_id})",
            f"Domain: {db_info.domain}",
            f"Confidence: {stage2_scores[selected_db_id]:.3f}"
        ]
        
        # Add stage 1 info
        similarity_score = next((score for db_id, score in stage1_candidates if db_id == selected_db_id), 0.0)
        explanation_parts.append(f"Schema similarity: {similarity_score:.3f}")
        
        # Add violations if any
        if violations:
            violation_summary = f"{len(violations)} policy considerations"
            explanation_parts.append(violation_summary)
        
        # Add alternatives
        alternatives = sorted(stage2_scores.items(), key=lambda x: x[1], reverse=True)[1:3]
        if alternatives:
            alt_text = ", ".join([f"{self.databases[db_id].name}({score:.3f})" for db_id, score in alternatives])
            explanation_parts.append(f"Alternatives: {alt_text}")
        
        return "; ".join(explanation_parts)
    
    def _log_routing_decision(self, query: str, context: QueryContext, result: RoutingResult) -> None:
        """Log routing decision for analysis and debugging."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user_id": context.user_id,
            "selected_db": result.selected_db_id,
            "confidence": result.confidence_score,
            "routing_time_ms": result.routing_time_ms,
            "candidates": len(result.stage1_candidates),
            "violations": len(result.policy_violations)
        }
        
        self.routing_history.append(log_entry)
        logger.info(f"Routed query to {result.selected_db_id} with confidence {result.confidence_score:.3f}")
    
    async def execute_query(
        self, 
        sql: str, 
        db_id: str, 
        context: QueryContext
    ) -> QueryExecutionResult:
        """
        Execute SQL query on selected database with validation and limits.
        
        Args:
            sql: SQL query to execute
            db_id: Target database identifier
            context: Query execution context
        
        Returns:
            Query execution result
        """
        start_time = time.time()
        
        try:
            # Validate SQL
            validation_result = self.sql_validator.validate_sql(sql, db_id)
            if not validation_result.is_valid:
                return QueryExecutionResult(
                    success=False,
                    sql=sql,
                    rows_returned=0,
                    execution_time_ms=0,
                    data=None,
                    error=f"SQL validation failed: {validation_result.error}",
                    warnings=validation_result.warnings
                )
            
            # Check policy violations
            violations = self.policy_engine.validate_sql_query(sql, db_id, context)
            critical_violations = [v for v in violations if v.suggested_action == PolicyAction.DENY]
            
            if critical_violations:
                error_msg = "; ".join([v.message for v in critical_violations])
                return QueryExecutionResult(
                    success=False,
                    sql=sql,
                    rows_returned=0,
                    execution_time_ms=0,
                    data=None,
                    error=f"Policy violation: {error_msg}",
                    warnings=[]
                )
            
            # Enforce query limits
            limited_sql = self.policy_engine.enforce_query_limits(sql, db_id)
            
            # Execute query (this would connect to actual database)
            # For now, return mock result
            execution_time = (time.time() - start_time) * 1000
            
            result = QueryExecutionResult(
                success=True,
                sql=limited_sql,
                rows_returned=100,  # Mock data
                execution_time_ms=execution_time,
                data=[{"mock": "data"}],  # Mock data
                error=None,
                warnings=[v.message for v in violations if v.severity in ["medium", "low"]]
            )
            
            # Update performance metrics
            self._update_performance_metrics(db_id, execution_time, True)
            
            # Log execution
            self.policy_engine.log_query_execution(context, limited_sql, violations, {
                "rows_returned": result.rows_returned,
                "execution_time_ms": result.execution_time_ms
            })
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(db_id, execution_time, False)
            
            logger.error(f"Query execution failed: {str(e)}")
            return QueryExecutionResult(
                success=False,
                sql=sql,
                rows_returned=0,
                execution_time_ms=execution_time,
                data=None,
                error=str(e),
                warnings=[]
            )
    
    def _update_performance_metrics(self, db_id: str, execution_time: float, success: bool) -> None:
        """Update performance metrics for a database."""
        if db_id not in self.performance_metrics:
            self.performance_metrics[db_id] = {
                "avg_query_time": 0.0,
                "success_rate": 1.0,
                "total_queries": 0,
                "last_response_time": 0.0
            }
        
        metrics = self.performance_metrics[db_id]
        
        # Update running averages
        total_queries = metrics["total_queries"]
        metrics["avg_query_time"] = (
            (metrics["avg_query_time"] * total_queries + execution_time) / (total_queries + 1)
        )
        
        successful_queries = metrics["success_rate"] * total_queries
        if success:
            successful_queries += 1
        metrics["success_rate"] = successful_queries / (total_queries + 1)
        
        metrics["total_queries"] = total_queries + 1
        metrics["last_response_time"] = execution_time
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and metrics."""
        total_routes = len(self.routing_history)
        if total_routes == 0:
            return {"total_routes": 0}
        
        # Calculate average routing time
        avg_routing_time = sum(r["routing_time_ms"] for r in self.routing_history) / total_routes
        
        # Database selection frequency
        db_selections = {}
        for route in self.routing_history:
            db_id = route["selected_db"]
            db_selections[db_id] = db_selections.get(db_id, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r["confidence"] for r in self.routing_history) / total_routes
        
        return {
            "total_routes": total_routes,
            "avg_routing_time_ms": avg_routing_time,
            "avg_confidence": avg_confidence,
            "database_selections": db_selections,
            "performance_metrics": self.performance_metrics
        }
    
    def get_database_list(self) -> List[Dict[str, Any]]:
        """Get list of registered databases with metadata."""
        return [
            {
                "db_id": db_info.db_id,
                "name": db_info.name,
                "domain": db_info.domain,
                "last_updated": db_info.last_updated.isoformat(),
                "metrics": self.performance_metrics.get(db_info.db_id, {})
            }
            for db_info in self.databases.values()
        ]
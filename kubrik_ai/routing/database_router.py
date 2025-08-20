"""
Database Router - Main intelligent routing logic that combines 
schema embeddings with classification and rules to select the optimal database.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re

from .schema_embedder import SchemaEmbedder, SchemaMetadata

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of SQL queries for classification"""
    SELECT = "select"
    ANALYTICAL = "analytical"
    JOINS = "joins"
    AGGREGATION = "aggregation"
    TIME_SERIES = "time_series"
    SEARCH = "search"


@dataclass
class RoutingDecision:
    """Result of database routing decision"""
    selected_database: str
    confidence_score: float
    reasoning: str
    similarity_scores: List[Tuple[str, float]]
    query_classification: QueryType
    fallback_databases: List[str]


@dataclass
class DatabaseConfig:
    """Configuration for a specific database"""
    database_id: str
    connection_string: str
    max_concurrent_queries: int = 10
    query_timeout_seconds: int = 300
    read_only: bool = True
    max_result_rows: int = 10000
    cost_weight: float = 1.0  # For cost-based routing
    performance_weight: float = 1.0  # For performance-based routing
    specializations: List[str] = None  # e.g., ["analytics", "real_time"]


class DatabaseRouter:
    """
    Intelligent database router that uses schema embeddings, query classification,
    and business rules to select the optimal database for a given query.
    """
    
    def __init__(self, schema_embedder: SchemaEmbedder):
        """
        Initialize the database router.
        
        Args:
            schema_embedder: Pre-trained schema embedder with database schemas
        """
        self.schema_embedder = schema_embedder
        self.database_configs: Dict[str, DatabaseConfig] = {}
        self.routing_rules: List[Dict[str, Any]] = []
        
    def add_database_config(self, config: DatabaseConfig) -> None:
        """Add configuration for a database."""
        self.database_configs[config.database_id] = config
        logger.info(f"Added configuration for database: {config.database_id}")
    
    def add_routing_rule(self, rule: Dict[str, Any]) -> None:
        """
        Add a routing rule for database selection.
        
        Rule format:
        {
            "name": "rule_name",
            "condition": {"query_type": "analytical", "domain": "finance"},
            "target_database": "analytics_db",
            "priority": 10
        }
        """
        self.routing_rules.append(rule)
        # Sort by priority (higher priority first)
        self.routing_rules.sort(key=lambda x: x.get('priority', 0), reverse=True)
        logger.info(f"Added routing rule: {rule['name']}")
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the type of SQL query based on keywords and patterns.
        
        Args:
            query: SQL query or natural language query
            
        Returns:
            Classified query type
        """
        query_lower = query.lower()
        
        # Time series patterns
        time_patterns = ['time', 'date', 'timestamp', 'hour', 'day', 'month', 'year', 
                        'trend', 'over time', 'historical']
        if any(pattern in query_lower for pattern in time_patterns):
            return QueryType.TIME_SERIES
            
        # Analytical patterns
        analytical_patterns = ['avg', 'average', 'sum', 'count', 'max', 'min', 
                             'group by', 'having', 'window', 'partition']
        if any(pattern in query_lower for pattern in analytical_patterns):
            return QueryType.ANALYTICAL
            
        # Join patterns
        join_patterns = ['join', 'inner join', 'left join', 'right join', 'full join']
        if any(pattern in query_lower for pattern in join_patterns):
            return QueryType.JOINS
            
        # Search patterns
        search_patterns = ['like', 'search', 'find', 'contains', 'match', 'similar']
        if any(pattern in query_lower for pattern in search_patterns):
            return QueryType.SEARCH
            
        # Aggregation patterns
        agg_patterns = ['total', 'summarize', 'aggregate', 'rollup']
        if any(pattern in query_lower for pattern in agg_patterns):
            return QueryType.AGGREGATION
            
        # Default to SELECT
        return QueryType.SELECT
    
    def apply_routing_rules(self, query: str, query_type: QueryType, 
                          candidate_databases: List[str]) -> Optional[str]:
        """
        Apply routing rules to select a database.
        
        Args:
            query: Original query
            query_type: Classified query type
            candidate_databases: List of candidate database IDs
            
        Returns:
            Selected database ID if a rule matches, None otherwise
        """
        for rule in self.routing_rules:
            condition = rule.get('condition', {})
            
            # Check query type condition
            if 'query_type' in condition:
                if condition['query_type'] != query_type.value:
                    continue
            
            # Check domain condition (based on schema metadata)
            if 'domain' in condition:
                domain_matches = []
                for db_id in candidate_databases:
                    metadata = self.schema_embedder.get_schema_metadata(db_id)
                    if metadata and metadata.domain == condition['domain']:
                        domain_matches.append(db_id)
                
                if not domain_matches:
                    continue
                candidate_databases = domain_matches
            
            # Check database specialization
            if 'specialization' in condition:
                spec_matches = []
                for db_id in candidate_databases:
                    config = self.database_configs.get(db_id)
                    if config and config.specializations:
                        if condition['specialization'] in config.specializations:
                            spec_matches.append(db_id)
                
                if not spec_matches:
                    continue
                candidate_databases = spec_matches
            
            # If we have a specific target database in the rule
            target_db = rule.get('target_database')
            if target_db and target_db in candidate_databases:
                logger.info(f"Applied routing rule '{rule['name']}' -> {target_db}")
                return target_db
            
            # If rule doesn't specify target, return first matching candidate
            if candidate_databases:
                logger.info(f"Applied routing rule '{rule['name']}' -> {candidate_databases[0]}")
                return candidate_databases[0]
        
        return None
    
    def calculate_database_score(self, database_id: str, query_type: QueryType, 
                               similarity_score: float) -> float:
        """
        Calculate a composite score for database selection.
        
        Args:
            database_id: Database identifier
            query_type: Type of query
            similarity_score: Schema similarity score
            
        Returns:
            Composite score for ranking
        """
        config = self.database_configs.get(database_id)
        if not config:
            return similarity_score
        
        # Base score from similarity
        score = similarity_score
        
        # Adjust based on database specializations
        if config.specializations:
            if query_type == QueryType.ANALYTICAL and "analytics" in config.specializations:
                score *= 1.2
            elif query_type == QueryType.TIME_SERIES and "time_series" in config.specializations:
                score *= 1.2
            elif query_type == QueryType.SEARCH and "search" in config.specializations:
                score *= 1.2
        
        # Adjust based on performance and cost weights
        score *= config.performance_weight
        score /= config.cost_weight  # Lower cost is better
        
        return score
    
    def route_query(self, query: str, top_k_candidates: int = 3) -> RoutingDecision:
        """
        Route a query to the optimal database.
        
        Args:
            query: Natural language or SQL query
            top_k_candidates: Number of candidate databases to consider
            
        Returns:
            Routing decision with selected database and reasoning
        """
        logger.info(f"Routing query: {query[:100]}...")
        
        # Step 1: Classify the query
        query_type = self.classify_query(query)
        logger.info(f"Query classified as: {query_type.value}")
        
        # Step 2: Get similar schemas using embeddings
        similarity_scores = self.schema_embedder.find_similar_schemas(
            query, top_k=top_k_candidates
        )
        
        if not similarity_scores:
            raise ValueError("No database schemas available for routing")
        
        # Step 3: Apply routing rules
        candidate_databases = [db_id for db_id, _ in similarity_scores]
        rule_selected = self.apply_routing_rules(query, query_type, candidate_databases)
        
        if rule_selected:
            # Find the similarity score for the rule-selected database
            rule_similarity = next(
                (score for db_id, score in similarity_scores if db_id == rule_selected),
                0.0
            )
            
            return RoutingDecision(
                selected_database=rule_selected,
                confidence_score=0.9,  # High confidence for rule-based selection
                reasoning=f"Selected by routing rule for {query_type.value} queries",
                similarity_scores=similarity_scores,
                query_classification=query_type,
                fallback_databases=[db_id for db_id, _ in similarity_scores[:3] if db_id != rule_selected]
            )
        
        # Step 4: Score-based selection if no rules apply
        scored_databases = []
        for db_id, similarity in similarity_scores:
            composite_score = self.calculate_database_score(db_id, query_type, similarity)
            scored_databases.append((db_id, composite_score))
        
        # Sort by composite score
        scored_databases.sort(key=lambda x: x[1], reverse=True)
        
        selected_db = scored_databases[0][0]
        confidence = scored_databases[0][1]
        
        reasoning = f"Selected based on similarity ({similarity_scores[0][1]:.3f}) and query type ({query_type.value})"
        
        return RoutingDecision(
            selected_database=selected_db,
            confidence_score=confidence,
            reasoning=reasoning,
            similarity_scores=similarity_scores,
            query_classification=query_type,
            fallback_databases=[db_id for db_id, _ in scored_databases[1:4]]
        )
    
    def get_database_config(self, database_id: str) -> Optional[DatabaseConfig]:
        """Get configuration for a specific database."""
        return self.database_configs.get(database_id)
    
    def list_available_databases(self) -> List[str]:
        """List all available database IDs."""
        return list(self.database_configs.keys())
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about the routing configuration."""
        return {
            'total_databases': len(self.database_configs),
            'total_rules': len(self.routing_rules),
            'database_types': list(set(
                self.schema_embedder.get_schema_metadata(db_id).database_type
                for db_id in self.database_configs.keys()
                if self.schema_embedder.get_schema_metadata(db_id)
            )),
            'specializations': list(set(
                spec for config in self.database_configs.values()
                if config.specializations
                for spec in config.specializations
            ))
        }
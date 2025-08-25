"""
KubrikAI: The Language Model for Data

Intelligent SQL routing and generation with advanced database understanding.
"""

__version__ = "0.1.0"
__author__ = "KubrikAI Team"

from .core.router import DatabaseRouter
from .core.schema_engine import SchemaEngine
from .core.policy_engine import PolicyEngine

__all__ = ["DatabaseRouter", "SchemaEngine", "PolicyEngine"]
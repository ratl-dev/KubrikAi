"""
Simple CLI test that demonstrates the CLI functionality without external dependencies
"""

import json
import tempfile
import os

# Create a sample database configuration
sample_db_config = {
    "type": "postgres",
    "connection_string": "postgresql://localhost/test",
    "tables": ["users", "orders", "products"],
    "columns": {
        "users": ["id", "name", "email", "created_at"],
        "orders": ["id", "user_id", "total", "order_date"],
        "products": ["id", "name", "price", "category"]
    },
    "relationships": [
        {
            "from_table": "orders",
            "from_column": "user_id", 
            "to_table": "users",
            "to_column": "id"
        }
    ],
    "description": "Test e-commerce database",
    "domain": "ecommerce",
    "read_only": True,
    "max_result_rows": 10000,
    "specializations": ["transactional"]
}

# Write to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(sample_db_config, f, indent=2)
    config_file = f.name

print(f"Created sample database config: {config_file}")
print("\nConfig contents:")
print(json.dumps(sample_db_config, indent=2))

print(f"\nTo test the CLI, you would run:")
print(f"kubrik-ai add-database test_db {config_file}")
print(f"kubrik-ai list-databases")
print(f"kubrik-ai route 'Show me all users with recent orders'")
print(f"kubrik-ai recommend 'Find customers by purchase history'")
print(f"kubrik-ai status")

# Clean up
os.unlink(config_file)
print(f"\nTemporary config file cleaned up.")
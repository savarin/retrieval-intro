"""
Main script for a SQL query generation system based on natural language input.

This script initializes the system, prepares sample data, and runs an
interactive loop for processing user queries.
"""

import json
from agent import Agent
from retrieve import VectorDB

if __name__ == "__main__":
    # Sample table definitions
    tables = [
        {
            "table_name": "dim_user",
            "columns": [
                {
                    "column_name": "user_id",
                    "column_type": "int",
                },
                {
                    "column_name": "user_name",
                    "column_type": "str",
                },
            ],
        },
        {
            "table_name": "fact_events",
            "columns": [
                {
                    "column_name": "event_id",
                    "column_type": "int",
                },
                {
                    "column_name": "event_name",
                    "column_type": "str",
                },
            ],
        },
    ]

    # Initialize the agent (handles embedding and SQL generation)
    print("Initializing agent...")
    agent = Agent()

    # Initialize the vector database (stores and retrieves embedded tables)
    print("Initializing vector db...")
    vector_db = VectorDB()

    # Embed and store each table in the vector database
    print("Inserting tables...")
    for table in tables:
        embedding_vector = agent.embed(json.dumps(table))
        vector_db.insert(table, embedding_vector)
        print(f"  {table['table_name']} done...")

    # Main interactive loop
    while True:
        # Get user input
        user_prompt = input("\n> ")

        # Exit condition
        if user_prompt == "exit":
            break

        # Process the user's query
        embedded_user_prompt = agent.embed(user_prompt)
        table = vector_db.get(embedded_user_prompt)
        sql = agent.codegen(table, user_prompt)

        # Display the generated SQL query
        print(f"\nSQL: {sql}")

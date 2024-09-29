"""
Main script for a SQL query generation system based on natural language input.

This script initializes the system, prepares sample data, and runs an
interactive loop for processing user queries.
"""

from typing import Tuple
import json

from agent import Agent, convert_text_to_embedding_vector
from retrieve import VectorDB


# Sample table definitions
TABLES = [
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


def set_up() -> Tuple[Agent, VectorDB]:
    """
    Set up the SQL query generation system by initializing the Agent and
    VectorDB, and populating the VectorDB with table embeddings.

    This function performs the following steps:
    1. Initializes an Agent for embedding and SQL generation.
    2. Initializes a VectorDB for storing and retrieving embedded tables.
    3. Embeds each predefined table and stores it in the VectorDB.

    Returns:
        Tuple[Agent, VectorDB]: A tuple containing the initialized Agent and
        VectorDB.
    """
    # Initialize the agent (handles embedding and SQL generation)
    print("Initializing agent...")
    agent = Agent()

    # Initialize the vector database (stores and retrieves embedded tables)
    print("Initializing vector db...")
    vector_db = VectorDB()

    # Embed and store each table in the vector database
    print("Inserting tables...")
    for table in TABLES:
        embedding_vector = convert_text_to_embedding_vector(json.dumps(table))
        vector_db.insert(table, embedding_vector)
        print(f"  {table['table_name']} done...")

    return agent, vector_db


if __name__ == "__main__":
    agent, vector_db = set_up()

    # Main interactive loop
    while True:
        # Get user input
        user_prompt = input("\n> ")

        # Exit condition
        if user_prompt == "exit":
            break

        # Process the user query
        embedded_user_prompt = convert_text_to_embedding_vector(user_prompt)
        table = vector_db.get(embedded_user_prompt)
        sql = agent.generate_sql(table, user_prompt)

        # Display the generated SQL query
        print(f"\nSQL: {sql}")

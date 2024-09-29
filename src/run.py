"""
Main script for a SQL query generation system based on natural language input.

This script initializes the system, prepares sample data, and runs an
interactive loop for processing user queries.
"""

from typing import Tuple
import json

from agent import Agent
from vector_db import VectorDB


# Sample table definitions
TABLES = [
    {
        "table_name": "dim_passengers",
        "columns": [
            {
                "column_name": "passenger_id",
                "column_type": "int",
                "column_description": "Unique identifier for each passenger",
            },
            {
                "column_name": "ticket_id",
                "column_type": "int",
                "column_description": "Identifier for the ticket associated with the passenger",
            },
            {
                "column_name": "passenger_name",
                "column_type": "str",
                "column_description": "Full name of the passenger",
            },
            {
                "column_name": "sex",
                "column_type": "str",
                "column_description": "Gender of the passenger (male or female)",
            },
            {
                "column_name": "age",
                "column_type": "float",
                "column_description": "Age of the passenger in years",
            },
            {
                "column_name": "is_survivor",
                "column_type": "bool",
                "column_description": "Indicates whether the passenger survived (True) or not (False)",
            },
        ],
    },
    {
        "table_name": "dim_tickets",
        "columns": [
            {
                "column_name": "ticket_id",
                "column_type": "int",
                "column_description": "Unique identifier for each ticket",
            },
            {
                "column_name": "port_id",
                "column_type": "str",
                "column_description": "Port of embarkation (S: Southampton, C: Cherbourg, Q: Queenstown)",
            },
            {
                "column_name": "code",
                "column_type": "str",
                "column_description": "Ticket code or number",
            },
            {
                "column_name": "fare",
                "column_type": "float",
                "column_description": "Price paid for the ticket",
            },
            {
                "column_name": "cabin",
                "column_type": "str",
                "column_description": "Cabin number (may be null)",
            },
        ],
    },
    {
        "table_name": "dim_ports",
        "columns": [
            {
                "column_name": "port_id",
                "column_type": "str",
                "column_description": "Unique identifier for each port (C: Cherbourg, Q: Queenstown, S: Southampton)",
            },
            {
                "column_name": "port_name",
                "column_type": "str",
                "column_description": "Full name of the port",
            },
            {
                "column_name": "country",
                "column_type": "str",
                "column_description": "Country where the port is located",
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
        vector_db.insert(json.dumps(table))
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
        table = json.loads(vector_db.get_top_k(user_prompt, 1)[0][1])
        sql = agent.generate_sql(table, user_prompt)

        # Display the generated SQL query
        print(f"\nSQL: {sql}")

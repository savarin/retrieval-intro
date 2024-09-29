"""
Evaluation script for the SQL query generation system.

This script defines a set of evaluation cases and runs them against the SQL
query generation model to assess its performance.
"""

from typing import List
from dataclasses import dataclass
import json

from .run import set_up


@dataclass
class Eval:
    """
    Represents an individual evaluation.

    Attributes:
        user_prompt (str): The natural language query input by the user.
        target_codes (List[str]): A list of acceptable SQL queries for this
        prompt.
    """

    user_prompt: str
    target_codes: List[str]


# Define a list of evaluation cases
EVALS = [
    Eval(
        user_prompt="Number of passengers",
        target_codes=[
            "SELECT COUNT(passenger_id) FROM dim_passengers;",
            "SELECT COUNT(*) FROM dim_passengers;",
        ],
    ),
    Eval(
        user_prompt="Number of passengers above 21 years of age",
        target_codes=[
            "SELECT COUNT(passenger_id) FROM dim_passengers WHERE age > 21;",
            "SELECT COUNT(*) FROM dim_passengers WHERE age > 21;",
        ],
    ),
]


if __name__ == "__main__":
    # Set up the agent and vector database
    agent, vector_db = set_up()

    success_count = 0
    print("Running evals...")

    # Iterate through each evaluation case
    for individual_eval in EVALS:
        # Process the user query
        top_1 = vector_db.get_top_k(individual_eval.user_prompt, 1)
        tables = [json.loads(pair[1]) for pair in top_1]
        sql = agent.generate_sql(tables, individual_eval.user_prompt)

        # Check if the generated SQL matches any of the target codes
        if sql in individual_eval.target_codes:
            success_count += 1

    # Print the final success ratio
    print(f"\nSuccess ratio: {success_count} / {len(EVALS)}")

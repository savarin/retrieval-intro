"""
Evaluation script for the SQL query generation system.

This script defines a set of evaluation cases and runs them against the SQL
query generation model to assess its performance.
"""

from typing import List
from dataclasses import dataclass

from agent import convert_text_to_embedding_vector
from run import set_up


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
        user_prompt="Number of users",
        target_codes=[
            "SELECT COUNT(user_id) FROM dim_user;",
            "SELECT COUNT(*) FROM dim_user;",
        ],
    ),
    Eval(
        user_prompt="Number of users with id > 100",
        target_codes=[
            "SELECT COUNT(user_id) FROM dim_user WHERE user_id > 100;",
            "SELECT COUNT(*) FROM dim_user WHERE user_id > 100;",
        ],
    ),
    Eval(
        user_prompt="Number of events",
        target_codes=[
            "SELECT COUNT(event_id) FROM fact_events;",
            "SELECT COUNT(*) FROM fact_events;",
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
        embedded_user_prompt = convert_text_to_embedding_vector(
            individual_eval.user_prompt
        )
        table = vector_db.get(embedded_user_prompt)
        sql = agent.codegen(table, individual_eval.user_prompt)

        # Check if the generated SQL matches any of the target codes
        if sql in individual_eval.target_codes:
            success_count += 1

    # Print the final success ratio
    print(f"\nSuccess ratio: {success_count} / {len(EVALS)}")

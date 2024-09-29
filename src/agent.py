"""
Defines the Agent class for embedding and SQL generation.

This module contains the core functionality for processing user queries and
generating SQL. It uses OpenAI's API for both text embedding and SQL generation.
"""

from typing import Dict, List
from dataclasses import dataclass
import json
import os

from openai import OpenAI
from pydantic import BaseModel

from retrieve import Table


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def convert_text_to_embedding_vector(text: str) -> List[float]:
    """
    Create an embedding vector from the given text string.

    Args:
        text (str): The input text string to embed.

    Returns:
        List[float]: The embedding vector.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    return (
        client.embeddings.create(
            input=[text],
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )


@dataclass
class Response(BaseModel):
    """Pydantic model for the expected response format from the OpenAI API."""

    sql: str


def create_template(table: str, query: str) -> List[Dict[str, str]]:
    """
    Create a template for the chat completion API.

    Args:
        table (str): JSON string representation of the table structure.
        query (str): User's natural language query.

    Returns:
        List[Dict[str, str]]: A list of message dictionaries for the chat
        completion API.
    """
    return [
        {
            "role": "system",
            "content": f"""\
You are a world-class data analyst. Please return the minimum SQL to the
user request and do not use aliases.

The relevant table is: {table}""",
        },
        {
            "role": "user",
            "content": query,
        },
    ]


@dataclass
class Agent:
    """
    Agent class for handling embedding and SQL generation tasks.
    """

    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def codegen(self, table: Table, user_prompt: str) -> str:
        """
        Generate SQL code based on the table structure and user prompt.

        Args:
            table (Table): The table structure.
            user_prompt (str): The user's natural language query.

        Returns:
            str: The generated SQL query.
        """
        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=create_template(json.dumps(table), user_prompt),
            response_format=Response,
        )
        message = completion.choices[0].message.parsed
        assert message is not None
        return message.sql

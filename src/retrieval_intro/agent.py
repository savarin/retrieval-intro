"""
Defines the Agent class for embedding and SQL generation.

This module contains the core functionality for processing user queries and
generating SQL. It uses OpenAI's API for both text embedding and SQL generation.
"""

from typing import List, Optional, TypedDict
from dataclasses import dataclass
import json
import os

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel


# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def convert_text_to_embedding_vector(
    text: str, api_key: Optional[str] = None
) -> List[float]:
    """
    Create an embedding vector from the given text string.

    Args:
        text (str): The input text string to embed.
        api_key (Optional[str]): OpenAI API key.

    Returns:
        List[float]: The embedding vector.
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    return (
        client.embeddings.create(
            input=[text],
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )


class Column(TypedDict):
    """Represents a column in a database table."""

    column_name: str
    column_type: str
    column_description: str


class Table(TypedDict):
    """Represents a database table structure."""

    table_name: str
    columns: List[Column]


@dataclass
class Response(BaseModel):
    """Pydantic model for the expected response format from the OpenAI API."""

    sql: str


# def create_template(tables: List[Table], query: str) -> List[Dict[str, str]]:
def create_template(
    tables: List[Table], query: str
) -> List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam]:
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

The tables in descending order of relevance are: {json.dumps(tables)}""",
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

    def __post_init__(self, api_key: Optional[str] = None) -> None:
        """Initialize the OpenAI client."""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_sql(self, tables: List[Table], user_prompt: str) -> str:
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
            messages=create_template(tables, user_prompt),
            response_format=Response,
        )
        message = completion.choices[0].message.parsed
        assert message is not None
        return message.sql

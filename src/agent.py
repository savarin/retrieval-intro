from typing import Dict, List
from dataclasses import dataclass
import json
import os

from openai import OpenAI
from pydantic import BaseModel

from retrieve import Table


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Response(BaseModel):
    sql: str


def create_template(table: str, query: str) -> List[Dict[str, str]]:
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
    def __post_init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, string: str) -> List[float]:
        return (
            self.client.embeddings.create(
                input=[string],
                model="text-embedding-3-small",
            )
            .data[0]
            .embedding
        )

    def codegen(self, table: Table, user_prompt: str) -> str:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=create_template(json.dumps(table), user_prompt),
            response_format=Response,
        )
        message = completion.choices[0].message.parsed
        assert message is not None

        return message.sql

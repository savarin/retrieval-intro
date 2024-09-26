from dataclasses import dataclass
import json
import os

from openai import OpenAI
from pydantic import BaseModel


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Response(BaseModel):
    sql_query: str


def create_template(table, query):
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
    def __post_init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, string):
        return (
            self.client.embeddings.create(
                input=[string],
                model="text-embedding-3-small",
            )
            .data[0]
            .embedding
        )

    def codegen(self, table, user_prompt):
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=create_template(json.dumps(table), user_prompt),
            response_format=Response,
        )
        return completion.choices[0].message.parsed.sql_query

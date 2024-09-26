import json
import os

from openai import OpenAI
from pydantic import BaseModel

from retrieve import tables


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Response(BaseModel):
    sql_query: str


def create_messages(table, query):
    return [
        {
            "role": "system",
            "content": f"""\
You are a world-class data analyst. Please return the minimum SQL to the
user request and do not use aliases.

The relevant table is: {json.dumps(table)}""",
        },
        {
            "role": "user",
            "content": query,
        },
    ]


if __name__ == "__main__":
    while True:
        user_prompt = input("> ")

        if user_prompt == "exit":
            break

        if len(user_prompt) == 0:
            user_prompt = "Number of users."

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=create_messages(tables[0], user_prompt),
            response_format=Response,
        )
        print(completion.choices[0].message.parsed.sql_query)

import json
import os

from openai import OpenAI
from pydantic import BaseModel


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Response(BaseModel):
    sql_query: str


table = {
    "table_name": "dim_user",
    "columns": [
        {
            "column_name": "user_id",
            "column_type": "str",
        },
    ],
}


def create_messages(table, query):
    return [
        {
            "role": "system",
            "content": f"""\
    You are a world-class data analyst. Please return the minimum SQL to the
    user request. The relevant table is: {json.dumps(table)}""",
        },
        {
            "role": "user",
            "content": query,
        },
    ]


if __name__ == "__main__":
    while True:
        user_prompt = input("> ")

        if user_prompt == "exit" or len(user_prompt) == 0:
            break

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=create_messages(table, user_prompt),
            response_format=Response,
        )
        print(completion.choices[0].message.parsed)

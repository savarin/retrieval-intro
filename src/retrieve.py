from typing import List, TypedDict
import json
import os

from dataclasses import dataclass
from scipy.spatial.distance import cosine

from openai import OpenAI
from pydantic import BaseModel


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def embed_string(string):
    return (
        client.embeddings.create(
            input=[string],
            model="text-embedding-3-small",
        )
        .data[0]
        .embedding
    )


class Column(TypedDict):
    column_name: str
    column_type: str


class Table(TypedDict):
    table_name: str
    columns: List[Column]


class Element(TypedDict):
    table: Table
    embedded_table_vector: List[float]


@dataclass
class VectorDB:
    def __post_init__(self):
        self.elements: List[Tuple[Table, List[float]]] = []

    def insert(self, table, embedding_vector):
        element = {"table": table, "embedding_vector": embedding_vector}
        self.elements.append(element)


if __name__ == "__main__":
    tables = [
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

    print("Initializing vector db...")
    vector_db = VectorDB()

    print("Inserting tables...")
    for table in tables:
        embedding_vector = embed_string(json.dumps(table))
        vector_db.insert(table, embedding_vector)
        print(f"  {table['table_name']} done...")

    user_prompt = "Number of users."
    embedded_user_prompt = embed_string(user_prompt)

    print("Comparing distances...")
    for element in vector_db.elements:
        distance = cosine(element["embedding_vector"], embedded_user_prompt)
        print(f"  {element['table']['table_name']}: {distance}")

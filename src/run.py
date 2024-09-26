import json

from agent import Agent
from retrieve import VectorDB


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

    print("Initializing agent...")
    agent = Agent()

    print("Initializing vector db...")
    vector_db = VectorDB()

    print("Inserting tables...")
    for table in tables:
        embedding_vector = agent.embed(json.dumps(table))
        vector_db.insert(table, embedding_vector)
        print(f"  {table['table_name']} done...")

    while True:
        user_prompt = input("\n> ")

        if user_prompt == "exit":
            break

        embedded_user_prompt = agent.embed(user_prompt)
        table = vector_db.get(embedded_user_prompt)

        sql = agent.codegen(table, user_prompt)
        print(f"\nSQL: {sql}")

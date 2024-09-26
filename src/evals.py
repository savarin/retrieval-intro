from typing import List
from dataclasses import dataclass

from run import set_up


@dataclass
class Eval:
    user_prompt: str
    target_codes: List[str]


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
    agent, vector_db = set_up()
    success_count = 0

    print("Running evals...")
    for individual_eval in EVALS:
        embedded_user_prompt = agent.embed(individual_eval.user_prompt)
        table = vector_db.get(embedded_user_prompt)
        sql = agent.codegen(table, individual_eval.user_prompt)

        if sql in individual_eval.target_codes:
            success_count += 1

    print(f"\nSuccess ratio: {success_count} / {len(EVALS)}")

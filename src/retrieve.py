from typing import List, TypedDict

from dataclasses import dataclass
from scipy.spatial.distance import cosine


class Column(TypedDict):
    column_name: str
    column_type: str


class Table(TypedDict):
    table_name: str
    columns: List[Column]


class EmbeddedTable(TypedDict):
    table: Table
    embedding_vector: List[float]


@dataclass
class VectorDB:
    def __post_init__(self) -> None:
        self.embedded_tables: List[EmbeddedTable] = []

    def insert(self, table: Table, embedding_vector: List[float]) -> None:
        embedded_table: EmbeddedTable = {
            "table": table,
            "embedding_vector": embedding_vector,
        }
        self.embedded_tables.append(embedded_table)

    def get(self, query_vector: List[float]) -> Table:
        min_distance, table = None, None

        print("Comparing distances...")
        for embedded_table in self.embedded_tables:
            distance = cosine(embedded_table["embedding_vector"], query_vector)
            print(f"  {embedded_table['table']['table_name']}: {round(distance, 3)}")

            if min_distance is None or distance < min_distance:
                min_distance = distance
                table = embedded_table["table"]

        assert table is not None
        return table

"""
Defines the VectorDB class for storing and retrieving embedded tables.

This module implements a simple vector database for storing table structures
along with their embedding vectors, and retrieving the most similar table
based on a query vector.
"""

from typing import List, TypedDict
from dataclasses import dataclass

from scipy.spatial.distance import cosine


class Column(TypedDict):
    """Represents a column in a database table."""

    column_name: str
    column_type: str


class Table(TypedDict):
    """Represents a database table structure."""

    table_name: str
    columns: List[Column]


class EmbeddedTable(TypedDict):
    """Represents a table along with its embedding vector."""

    table: Table
    embedding_vector: List[float]


@dataclass
class VectorDB:
    """
    A simple vector database for storing and retrieving embedded tables.
    """

    def __post_init__(self) -> None:
        """
        Initialize the vector database with an empty list of embedded tables.
        """
        self.embedded_tables: List[EmbeddedTable] = []

    def insert(self, table: Table, embedding_vector: List[float]) -> None:
        """
        Insert a new table and its embedding vector into the database.

        Args:
            table (Table): The table structure to insert.
            embedding_vector (List[float]): The embedding vector of the table.
        """
        embedded_table: EmbeddedTable = {
            "table": table,
            "embedding_vector": embedding_vector,
        }
        self.embedded_tables.append(embedded_table)

    def get(self, query_vector: List[float]) -> Table:
        """
        Retrieve the most similar table to the given query vector.

        This method calculates the cosine distance between the query vector
        and each stored table's embedding vector, returning the table with
        the smallest distance (i.e. most similar).

        Args:
            query_vector (List[float]): The query vector to compare against.

        Returns:
            Table: The most similar table to the query vector.

        Raises:
            AssertionError: If no tables are found in the database.
        """
        min_distance, table = None, None
        print("Comparing distances...")
        for embedded_table in self.embedded_tables:
            # Calculate cosine distance between query and table vectors
            distance = cosine(embedded_table["embedding_vector"], query_vector)
            print(f"  {embedded_table['table']['table_name']}: {round(distance, 3)}")

            # Update if this is the closest match so far
            if min_distance is None or distance < min_distance:
                min_distance = distance
                table = embedded_table["table"]

        assert table is not None, "No tables found in the database"
        return table

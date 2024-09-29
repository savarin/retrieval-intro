"""
Defines the VectorDB class for storing and retrieving text-vector pairs.

This module implements a simple vector database for storing text along with
their embedding vectors, and retrieving the most similar text based on a query
vector.
"""

from typing import List, Tuple, TypedDict
from dataclasses import dataclass

from scipy.spatial.distance import cosine

from agent import convert_text_to_embedding_vector


class TextVectorPair(TypedDict):
    """Represents a text string along with its embedding vector."""

    text: str
    embedding_vector: List[float]


@dataclass
class VectorDB:
    """
    A simple vector database for storing and retrieving text-vector pairs.
    """

    def __post_init__(self) -> None:
        """
        Initialize the vector database with an empty list of pairs.
        """
        self.text_vector_pairs: List[TextVectorPair] = []

    def insert(self, text: str) -> None:
        """
        Convert text into embedding vector and insert a pair into the database.

        Args:
            text (str): The text to insert.
        """
        text_vector_pair: TextVectorPair = {
            "text": text,
            "embedding_vector": convert_text_to_embedding_vector(text),
        }
        self.text_vector_pairs.append(text_vector_pair)

    def get_top_k(self, query: str, k: int = 1) -> List[Tuple[float, str]]:
        """
        Retrieve the top-k most similar pairs to the given query string.

        This method calculates the Euclidean distance between the embeddings of
        the query string and each stored text string, returning the stored texts
        with the smallest distances.

        Args:
            query (str): The query string to compare against.

        Returns:
            List[TextVectorPair]: The most similar text-vector pairs.

        Raises:
            AssertionError: If no pairs are found in the database.
        """
        query_vector = convert_text_to_embedding_vector(query)
        distance_pairs = []

        for text_vector_pair in self.text_vector_pairs:
            # Calculate cosine distance between query and stored text
            distance = cosine(text_vector_pair["embedding_vector"], query_vector)
            distance_pairs.append((distance, text_vector_pair["text"]))

        return sorted(distance_pairs)[:k]

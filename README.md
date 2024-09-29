# retrieval-intro

This project implements a natural language to SQL query generation system using OpenAI's language models and vector embeddings. It allows users to input queries in natural language and receive corresponding SQL queries based on the available database schema.

## Components

The system consists of three main Python scripts:

1. `run.py`: The main script that orchestrates the entire process.
2. `agent.py`: Defines the `Agent` class for handling embedding and SQL generation tasks.
3. `vector_db.py`: Implements the `VectorDB` class for storing and retrieving text-vector pairs.

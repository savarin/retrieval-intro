# retrieval-intro

This project implements a natural language to SQL query generation system using OpenAI's language models and vector embeddings. It allows users to input queries in natural language and receive corresponding SQL queries based on the available database schema.

## Components

The system consists of three main Python scripts:

1. `run.py`: The main script that orchestrates the entire process.
2. `agent.py`: Defines the `Agent` class for handling embedding and SQL generation tasks.
3. `vector_db.py`: Implements the `VectorDB` class for storing and retrieving text-vector pairs.

## Quickstart

Install the required dependencies:
```
pip install -r requirements.txt
```

Create an `.env` file in the project root directory with your OpenAI API key (placeholder below):
```shell
OPENAI_API_KEY=sk-...
```

Set the environment variable:
```shell
export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d '=' -f2)
```

To run in interactive mode:
```python
python src/retrieval_intro/run.py
```

To run basic evals:
```python
python src/retrieval_intro/evals.py
```

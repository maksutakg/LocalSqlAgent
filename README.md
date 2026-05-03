# Enterprise Local SQL Agent 🤖🗄️

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-FF4F00?style=for-the-badge&logo=langchain)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Ollama-White?style=for-the-badge&logo=ollama&logoColor=black)](https://ollama.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)

Local SQL Agent built with **LangGraph**, **Ollama (Llama 3.1)**, **FastAPI**, and **SQLAlchemy**. This agent translates natural language queries into secure, read-only SQL commands, executes them asynchronously on your database, and explains the results back to the user in natural language—all running locally with zero data sent to external APIs like OpenAI.

## ✨ Features

- **Semantic Schema RAG:** Instantly scales to massive enterprise databases with hundreds of tables. Uses **FAISS** and **HuggingFace Embeddings** to retrieve only the top-K relevant table schemas to avoid LLM context-window exhaustion.
- **Value-Based Retrieval (Categorical RAG):** Automatically embeds categorical data (VARCHAR/TEXT values) into a vector store. When a user asks for "Laptops", the LLM explicitly knows which exact string matches exist in the database.
- **Tool Calling:** Enforces strict Pydantic JSON schemas via `bind_tools` to ensure generated SQL is clean and executable, abandoning brittle string-manipulation techniques.
- **Asynchronous Architecture:** Built from the ground up using `asyncpg`, `aiosqlite`, `aiomysql`, and LangGraph's `.ainvoke()`. Non-blocking architecture ready to handle hundreds of concurrent API requests.
- **Enterprise Security (Read-Only Guardrails):** Every LLM-generated SQL query is strictly validated. Non-SELECT queries (DROP, DELETE, UPDATE) are blocked before execution. As a final physical barrier, queries run within an asynchronous transaction that is ALWAYS rolled back (`await trans.rollback()`).
- **Anti-Hallucination Measures:** Strict LLM prompting ensures the agent never invents data. If SQL execution fails consecutively, the agent elegantly bypasses the LLM to return a hardcoded, safe error message.
- **Multi-Tenant Memory Caching:** Secure dictionary-based caching mechanism to isolate RAG vectorstores across different database connections.

## 🚀 Quick Start

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) (Extremely fast Python package installer)
- [Ollama](https://ollama.com/) installed and running locally with the Llama 3.1 model:
  ```bash
  ollama pull llama3.1
  ```

### 2. Setup
Clone the repository, install dependencies, and define your database connection:

```bash
uv sync
```

```bash
# Create a .env file and add your database URL
echo 'DATABASE_URL="postgresql://user:password@localhost:5432/your_database"' > .env
```

### 3. (Optional) Generate Mock Data
Don't have a database? We've got you covered. Run the standalone script to populate an SQLite or PostgreSQL database with hundreds of realistic dummy records (Users, Products, Orders) using Faker.

```bash
uv run python create_mock_data.py
```

### 4. Run the Server
Launch the asynchronous FastAPI backend:
```bash
uv run uvicorn main:app --reload
```

## 📡 API Usage

The system automatically detects your database dialect from `.env`. Just send your query!

### 📊 Ask a Database Question
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "What are the top 5 best-selling products?",
           "session_id": "user_123"
         }'
```

### 💬 General Chat
The built-in **Intent Parser** distinguishes between database queries and casual conversation.
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Hello, how can you help me?",
           "session_id": "user_123"
         }'
```

## 📂 Project Structure

- `agent.py`: LangGraph state machine, Intent Parser, SQL Generator, Validator, and LLM nodes.
- `database.py`: SQLAlchemy async operations, Semantic Schema RAG, and Value RAG logic.
- `main.py`: FastAPI application, CORS configuration, and endpoints.
- `create_mock_data.py`: Standalone Faker script to seed dummy databases for testing.

## 🤝 License
MIT License

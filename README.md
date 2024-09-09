# Chatbot Service

This project implements a multi-user chatbot service using FastAPI, OpenAI's GPT-4o model, and LanceDB for hybrid search. The service supports API key-based authentication, chat history storage, and hybrid search with optional reranking.

## Chat Endpoint

```
https://chat-app-gdms.onrender.com/chat
```

## Architecture Overview

### Components

1. **FastAPI**: The web framework used to build the API endpoints.
2. **OpenAI**: The client used to interact with OpenAI's GPT-4o model.
3. **LanceDB**: The vector database used for hybrid search (BM25 + semantic search).
4. **CohereReranker**: An optional reranker model from Cohere used to improve search results.
5. **dotenv**: Used to load environment variables from a `.env` file.

### Key Features

- **Multi-User Design**: Each user is identified by a unique API key. Chat histories are stored separately for each user.
- **Chat History Storage**: Chat histories (Last ten messages) are stored in a JSON file with hashed API keys as identifiers.
- **Hybrid Search**: Combines BM25 and semantic search to find relevant context from a knowledge base.
- **Optional Reranking**: Uses Cohere's reranker model to improve search results.
- **Streaming Responses**: Supports streaming responses for real-time interaction.

## Setup

### Prerequisites

- Python 3.7+
- An OpenAI API key
- A Cohere API key (if using reranking)
- A `.env` file with the following variables:
  - `OPENAI_API_KEY`
  - `BEARER_TOKEN` (comma-separated list of valid API keys)
  - `COHERE_API_KEY` (if using reranking)

### Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/jashdalvi/chat-app.git
   cd chat-app
   ```

2. Install the dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Set up the `.env` file:

   ```sh
   touch .env
   # Edit .env to include your API keys
   ```

4. Set up the vector database:
   - Use Firecrawl to scrape `artisan.co` and index the chunks into the vector database located in the `db` directory.

### Running the Service

Start the FastAPI server:

```sh
uvicorn main:app --reload
```

## API Endpoints

### POST /chat

Handles user messages and returns responses from the chatbot.

#### Request Body

```json
{
  "message": "string",
  "rerank": "boolean (optional)",
  "stream": "boolean (optional)",
  "reset": "boolean (optional)"
}
```

#### Headers

- `x-api-key`: The API key for authentication.

#### Response

- Returns the chatbot's response or a streaming response if `stream` is set to `true`.

## Code Overview

### main.py

- **Imports**: Imports necessary libraries and modules.
- **ChatMessage**: Defines the request body schema.
- **Client Initialization**: Initializes the OpenAI client.
- **Database Setup**: Connects to the LanceDB database and sets up the reranker.
- **System Message**: Defines the system message for the chatbot.
- **Tools**: Defines the tools available to the chatbot.
- **Chat History Functions**: Functions to load and save chat history.
- **Search Function**: Function to search the knowledge base with optional reranking.
- **Chat Streamer**: Function to handle streaming responses.
- **API Key Dependency**: Function to validate API keys.
- **Hash API Key**: Function to hash API keys.
- **FastAPI App**: Initializes the FastAPI app and defines the `/chat` endpoint.

## Conclusion

This project demonstrates a robust architecture for a multi-user chatbot service with advanced search capabilities. By leveraging FastAPI, OpenAI, and LanceDB, it provides a scalable and efficient solution for customer service automation.

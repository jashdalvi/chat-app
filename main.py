from fastapi import FastAPI, Depends, HTTPException, Header
import os
from dotenv import load_dotenv
load_dotenv()
import hashlib
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json
import lancedb
from lancedb.rerankers import CohereReranker
from collections import deque
from fastapi.responses import StreamingResponse

# Request body schema
class ChatMessage(BaseModel):
    message: str
    rerank: Optional[bool] = False  # Rerank flag, default to False
    stream: Optional[bool] = False  # Stream flag, default to False
    reset: Optional[bool] = False  # Reset flag, default to False

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Open the database for hybrid search
uri = "./db"  # Define the URI for the database
db = lancedb.connect(uri)  # Connect to the LanceDB database using the specified URI
table = db.open_table("artisan")  # Open the "artisan" table in the database
reranker = CohereReranker()  # Initialize a reranker using Cohere's reranking model
base_path = "/var/data"  # Define the base path for storing data files

model = "gpt-4o"
system_message = (
    "You are a helpful assistant named Ava that helps with customer service. "
    "You work for a company named Artisan. You are a friendly and knowledgeable assistant "
    "that is always ready to help. Make sure you call the function search_knowledge_base to search "
    "the knowledge base for additional information. Also search the knowledge base when the user asks "
    "the question like 'what can you do?' or 'what do you know?'."
)

# Define tools for the assistant
tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Searches the knowledge base for additional information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for in the knowledge base.",
                },
            },
            "required": ["query"],
        },
    }
}]

# Load chat history from file
def load_chat_history():
    chat_history_path = os.path.join(base_path,"chat_history.json")
    if os.path.exists(chat_history_path):
        with open(chat_history_path, "r") as file:
            data = json.load(file)
            # Convert lists back to deques
            return {k: deque(v, maxlen=10) for k, v in data.items()}
    return {}

# Save chat history to file
def save_chat_history(chat_history):
    with open(os.path.join(base_path,"chat_history.json"), "w") as file:
        # Convert deques to lists for serialization
        json.dump({k: list(v) for k, v in chat_history.items()}, file)

# Initialize chat history
chat_history = load_chat_history()

# Function to search the knowledge base
def search_knowledge_base(query, rerank=False):
    # Search the knowledge base for the query
    if not rerank:
        return table.search(query, query_type="hybrid").limit(10).to_pandas()["text"].to_list()
    return table.search(query, query_type="hybrid").limit(10).rerank(reranker=reranker).to_pandas()["text"].to_list()


# Async generator to stream chat responses
async def chat_streamer(user_messages, chat_history):
    message_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_message,
            }
        ] + list(user_messages),
        temperature=0,
        stream=True
    )
    final_response = ""
    for chunk in message_response:
        if chunk.choices[0].delta.content:
            final_response += chunk.choices[0].delta.content
            yield chunk.choices[0].delta.content
    
    user_messages.append({"role": "assistant", "content": final_response})
    save_chat_history(chat_history)

# Dependency to check API key
def get_api_key(x_api_key: str = Header(...)):
    if x_api_key not in os.environ.get("BEARER_TOKEN").split(","):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# Hash the API key
def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

app = FastAPI()

@app.post("/chat")
async def chat(message: ChatMessage, api_key: str = Depends(get_api_key)):
    hashed_api_key = hash_api_key(api_key)
    
    if hashed_api_key not in chat_history or message.reset:
        chat_history[hashed_api_key] = deque(maxlen=10)

    user_messages = chat_history[hashed_api_key]
    # Handle edge case of first message being a tool call
    if len(user_messages) > 0 and user_messages[0]["role"] == "tool":
        user_messages.popleft()
    
    # Add the user message to the chat history
    user_messages.append({
        "role": "user",
        "content": message.message,
    })
    # Generate a query to search the knowledge base
    # this is helpful for follow-up questions and ambigious queries
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_message,
            }
        ] + list(user_messages),
        model=model,
        tools=tools,
        temperature=0,
        tool_choice={"type": "function", "function": {"name": "search_knowledge_base"}}
    )
    
    response_message = response.choices[0].message
    if response_message.tool_calls:
        # Add the response message to the chat history
        user_messages.append(response_message.dict())
        tool_call_id = response_message.tool_calls[0].id
        tool_function_name = response_message.tool_calls[0].function.name
        tool_query_string = json.loads(response_message.tool_calls[0].function.arguments)['query']
        # Call the tool function to search the knowledge base
        search_result = search_knowledge_base(tool_query_string, message.rerank)
        # Add the search result to the chat history
        user_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_function_name,
            "content": json.dumps(search_result)
        })
        
        # Stream the chat responses if the stream flag is set
        if not message.stream:
            message_response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    }
                ] + list(user_messages),
                temperature=0
            )
            user_messages.append({"role": "assistant", "content": message_response.choices[0].message.content})
            save_chat_history(chat_history)
            return message_response.choices[0].message.content
        else:
            return StreamingResponse(chat_streamer(user_messages, chat_history), media_type='text/event-stream')
    else:
        user_messages.append(response_message.dict())
        save_chat_history(chat_history)
        return response_message.content
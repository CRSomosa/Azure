import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch

app = FastAPI()

# Configure OpenAI
openai.api_base = os.getenv("GPT4_API_BASE")  # Matches your Azure env var
openai.api_key = os.getenv("GPT4_API_KEY")    # Matches your Azure env var
openai.api_type = "azure"
openai.api_version = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")  # fallback just in case

# Embeddings setup
embedding_deployment_name = os.getenv("EMBEDDINGS_API_NAME")  # Matches your Azure env var
embeddings = OpenAIEmbeddings(deployment=embedding_deployment_name, chunk_size=1)

# Azure Cognitive Search setup
acs = AzureSearch(
    azure_search_endpoint=os.getenv("SEARCH_SERVICE_NAME"),
    azure_search_key=os.getenv("SEARCH_API_KEY"),
    index_name=os.getenv("SEARCH_INDEX_NAME"),
    embedding_function=embeddings.embed_query
)

# Request Body
class Body(BaseModel):
    query: str

@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=301)

@app.post("/ask")
def ask(body: Body):
    """
    Use the query parameter to interact with Azure OpenAI Service
    using Azure Cognitive Search for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {"response": chat_bot_response}

def search(query):
    """
    Send the query to Azure Cognitive Search and return the top result.
    """
    docs = acs.similarity_search_with_relevance_scores(
        query=query,
        k=5,
    )
    result = docs[0][0].page_content
    print(result)
    return result

def assistant(query, context):
    messages = [
        # System prompt
        {
            "role": "system",
            "content": "Assistant is a chatbot that helps you find the best wine for your taste."
        },
        # User query
        {
            "role": "user",
            "content": query
        },
        # Context from vector search results
        {
            "role": "assistant",
            "content": context
        },
    ]

    # Use GPT4 deployment name from env var
    gpt_deployment_name = os.getenv("GPT4_DEPLOYMENT_NAME")  # Matches your Azure env var

    response = openai.ChatCompletion.create(
        engine=gpt_deployment_name,
        messages=messages,
    )

    return response["choices"][0]["message"]["content"]

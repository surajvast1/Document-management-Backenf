import openai
import os
import requests
import json
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Initialize environment variables and models
openai_key = os.getenv('OPENAI_API_KEY')
QDRANT_URL = os.getenv('QDRANT_URL')

embeddings_model = OpenAIEmbeddings(openai_api_key=openai_key)
client = QdrantClient(url=QDRANT_URL)

MAX_CONTEXT_TOKENS = 7500  # Maximum tokens to ensure it's within the model's limit

def generate_embedding_for_question(question):
    """Generates an embedding for the given question."""
    try:
        return embeddings_model.embed_documents([question])[0]
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def fetch_relevant_context(collection_name, question):
    """Fetches relevant context from Qdrant for a given question and truncates if necessary."""
    try:
        question_embedding = generate_embedding_for_question(question)
        
        # Search in Qdrant for relevant contexts
        search_results = client.search(
            collection_name=collection_name,
            query_vector=question_embedding
        )
        
        # Concatenate the context text
        context_text = " ".join(result.payload.get('text', 'No text found') for result in search_results)
        
        # Truncate the context text to fit within the token limit
        truncated_context = truncate_context(context_text, MAX_CONTEXT_TOKENS)
        
        return truncated_context
    except Exception as e:
        raise Exception(f"Error fetching relevant context: {str(e)}")

def truncate_context(text, max_tokens):
    """Truncates text to fit within the maximum token limit."""
    tokens = text.split()  # Simple tokenization by whitespace
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    return text

def get_ans(context, question):
    """Queries OpenAI API to generate a response based on context and question."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an advanced AI assistant. Use only the provided context to answer the question."
                    f" #Context#: {context}"
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
        }

        # Send the request to OpenAI
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error querying OpenAI API: {str(e)}")

def lambda_handler(event, context):
    """AWS Lambda handler to process requests and generate responses."""
    try:
        body = json.loads(event['body']) if 'body' in event and event['body'] else event
        collection_name = body['collection_name']
        question = body['question']
        
        relevant_context = fetch_relevant_context(collection_name, question)
        
        if not relevant_context:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"message": "No relevant context found for the question."})
            }

        response_message = get_ans(relevant_context, question)
        
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": response_message})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"message": str(e)})
        }

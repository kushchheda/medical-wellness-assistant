"""
Medical Wellness Assistant - Flask Application Entry Point

This module initializes the Flask app, sets up the RAG pipeline,
and defines the API routes for the chatbot interface.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import download_hugging_face_embeddings
from src.prompt import SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "chatbot")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError(
        "Missing required environment variables. "
        "Please ensure PINECONE_API_KEY and OPENAI_API_KEY are set in your .env file."
    )

# ---------------------------------------------------------------------------
# RAG Pipeline Setup
# ---------------------------------------------------------------------------

logger.info("Loading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()

logger.info(f"Connecting to Pinecone index: '{PINECONE_INDEX_NAME}'...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

logger.info("Initializing OpenAI LLM and prompt chain...")
llm = OpenAI(
    api_key=OPENAI_API_KEY,
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

logger.info("RAG pipeline ready.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main chatbot UI."""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """
    Handle a user message and return the assistant's response.

    Expects a form field 'msg' in the POST request body.
    Returns a plain-text response from the RAG chain.
    """
    user_message = request.form.get("msg", "").strip()

    if not user_message:
        return jsonify({"error": "No input received."}), 400

    logger.info(f"User message: {user_message}")

    try:
        result = rag_chain.invoke({"input": user_message})
        answer = (
            result.get("answer")
            or result.get("output")
            or "I'm sorry, I was unable to generate a response."
        )
        logger.info(f"Assistant response: {answer}")
    except Exception as e:
        logger.error(f"Error invoking RAG chain: {e}", exc_info=True)
        answer = "An error occurred while processing your request. Please try again."

    return str(answer)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

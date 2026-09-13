"""
LLM generation for the Aventro Motors RAG system.

Responsibility:
  - wrap Groq's chat model behind a small, stable interface
  - provide two ways to generate an answer given a query + context
    (structured prompt, and a simple fallback)

It does NOT:
  - load the .env file (that's the caller's job)
  - know anything about retrieval or ChromaDB
  - decide how many documents to retrieve (that's pipeline/)
"""
import os
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from rag.config import GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS


class GroqLLM:
    """Thin wrapper around Groq's chat model."""

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Groq LLM.

        Args:
            model_name: Groq model name.
            api_key: Groq API key. If None, read from GROQ_API_KEY env var.

        Raises:
            ValueError: if no API key is available.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )

        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str) -> str:
        """
        Generate a response using retrieved context (structured prompt).

        Args:
            query: User question.
            context: Retrieved document context.

        Returns:
            Generated response string (or an error message on failure).
        """
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so.""",
        )

        formatted_prompt = prompt_template.format(context=context, question=query)

        try:
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Generate a response with a minimal prompt (no role framing).

        Args:
            query: User question.
            context: Retrieved context.

        Returns:
            Generated response string (or an error message on failure).
        """
        simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""

        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
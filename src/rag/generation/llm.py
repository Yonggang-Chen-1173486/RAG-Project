"""
LLM generation for the Aventro Motors RAG system.
"""
import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from rag.config import GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

logger = logging.getLogger(__name__)


class GroqLLM:
    """Thin wrapper around Groq's chat model."""

    def __init__(
        self,
        model_name: str = GROQ_MODEL_NAME,
        api_key: Optional[str] = None,
    ):
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

        logger.info("Initialized Groq LLM with model: %s", self.model_name)

    def invoke(self, prompt: str) -> str:
        """Send a raw prompt string to the LLM and return the reply as a string."""
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return response.content
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

    def invoke(self, prompt: str, max_retries: int = 3) -> str:
        """
        Send a raw prompt string to the LLM and return the reply as a string.

        Retries on rate-limit (429) errors with exponential backoff.
        """
        import time

        messages = [HumanMessage(content=prompt)]

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2 ** attempt   # 2s, 4s
                    logger.warning("Rate limit hit; retrying in %ds...", wait)
                    time.sleep(wait)
                    last_error = e
                    continue
                raise
        raise last_error  # should not reach here
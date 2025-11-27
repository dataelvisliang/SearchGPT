import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterService:
    """Service for interacting with OpenRouter API"""

    def __init__(self, api_key: str, model_name: str = "x-ai/grok-4.1-fast:free"):
        """
        Initialize OpenRouter service

        Args:
            api_key: OpenRouter API key
            model_name: Model identifier (default: x-ai/grok-4.1-fast:free)
                       Available free models:
                       - "x-ai/grok-4.1-fast:free" (GPT-4 equivalent)
                       - "openai/gpt-oss-20b:free" (GPT-3.5 equivalent)
        """
        self.api_key = api_key
        self.model_name = model_name

    def call_openrouter(self, messages: list, temperature: float = 0.0, stream: bool = False) -> dict:
        """
        Call OpenRouter API for chat completion

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response

        Returns:
            Response dictionary with 'content' and 'role'
        """
        if not self.api_key:
            logger.error("OpenRouter API key is missing")
            raise ValueError("⚠️ OpenRouter API key is required. Get a free key at https://openrouter.ai/")

        logger.info(f"Calling OpenRouter API with model: {self.model_name}, stream: {stream}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "SearchGPT Application"
        }

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature
        }

        try:
            if stream:
                return self._stream_response(headers, payload)
            else:
                logger.debug("Sending non-streaming request to OpenRouter")
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()

                logger.info(f"OpenRouter API response received. Status: {response.status_code}")

                # Extract the assistant's message
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0]["message"]
                    content_length = len(message.get("content", ""))
                    logger.info(f"Successfully received response ({content_length} characters)")
                    return {
                        "content": message.get("content", ""),
                        "role": message.get("role", "assistant")
                    }
                else:
                    logger.error(f"Unexpected response format: {result}")
                    raise ValueError("Unexpected response format from OpenRouter API")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling OpenRouter API: {e}", exc_info=True)
            raise

    def _stream_response(self, headers: dict, payload: dict):
        """
        Stream response from OpenRouter API

        Args:
            headers: Request headers
            payload: Request payload

        Yields:
            Chunks of the response content
        """
        payload["stream"] = True

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()

            full_content = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue

            print()  # New line after streaming
            return {
                "content": full_content,
                "role": "assistant"
            }

        except requests.exceptions.RequestException as e:
            logging.error(f"Error streaming from OpenRouter API: {e}")
            raise


class OpenRouterEmbeddings:
    """Wrapper for OpenRouter embeddings (uses text-embedding-ada-002 compatible model)"""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-ada-002"):
        """
        Initialize OpenRouter embeddings

        Args:
            api_key: OpenRouter API key
            model: Embedding model identifier
        """
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list) -> list:
        """
        Embed a list of documents

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list:
        """
        Embed a single query text

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list:
        """
        Get embedding for a single text

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not self.api_key:
            raise ValueError("⚠️ OpenRouter API key is required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "SearchGPT Application"
        }

        # OpenRouter uses the same embeddings endpoint as OpenAI
        url = "https://openrouter.ai/api/v1/embeddings"
        payload = {
            "model": self.model,
            "input": text
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            else:
                raise ValueError("Unexpected response format from embeddings API")

        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting embeddings from OpenRouter: {e}")
            raise

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
            "X-Title": "RelevanceSearch Application"
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


class GiteeEmbeddings:
    """Wrapper for Gitee AI embeddings (uses BGE-M3 model)"""

    def __init__(self, api_key: str, model: str = "bge-m3"):
        """
        Initialize Gitee AI embeddings

        Args:
            api_key: Gitee AI API key
            model: Embedding model identifier (default: bge-m3)
        """
        from openai import OpenAI

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            base_url="https://ai.gitee.com/v1",
            api_key=api_key,
            default_headers={"X-Failover-Enabled": "true"}
        )
        logger.info(f"Initialized Gitee AI embeddings with model: {model}")

    def embed_documents(self, texts: list) -> list:
        """
        Embed a list of documents

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(texts)} documents using Gitee AI {self.model}")
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings with Gitee AI: {e}", exc_info=True)
            raise

    def embed_query(self, text: str) -> list:
        """
        Embed a single query text

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        logger.debug(f"Generating embedding for query using Gitee AI {self.model}")
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            logger.debug("Query embedding generated successfully")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding with Gitee AI: {e}", exc_info=True)
            raise


class OpenRouterEmbeddings:
    """Wrapper for OpenRouter embeddings (uses text-embedding-ada-002 compatible model)"""

    def __init__(self, api_key: str, model: str = "openai/text-embedding-3-small"):
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
        Embed a list of documents using batching for efficiency

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} documents with batching")
        embeddings = []

        # Process in batches to reduce API calls (OpenRouter supports batching)
        batch_size = 20  # OpenRouter can handle multiple inputs per request

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1

            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")

            # Get embeddings for entire batch in one API call
            batch_embeddings = self._get_batch_embedding(batch)
            embeddings.extend(batch_embeddings)

        logger.info(f"Successfully embedded {len(embeddings)} documents")
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

    def _get_batch_embedding(self, texts: list, max_retries: int = 3) -> list:
        """
        Get embeddings for a batch of texts in a single API call

        Args:
            texts: List of text strings to embed
            max_retries: Maximum number of retry attempts

        Returns:
            List of embedding vectors
        """
        if not self.api_key:
            raise ValueError("⚠️ OpenRouter API key is required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RelevanceSearch Application"
        }

        url = "https://openrouter.ai/api/v1/embeddings"
        payload = {
            "model": self.model,
            "input": texts  # Send list of texts for batch processing
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                timeout = 60 + (attempt * 30)
                logger.info(f"Batch embedding API call attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)")

                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                if "data" in result and len(result["data"]) > 0:
                    # Extract embeddings in order
                    return [item["embedding"] for item in result["data"]]
                else:
                    raise ValueError("Unexpected response format from embeddings API")

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Timeout on batch attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Request error on batch attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

        logger.error(f"All {max_retries} batch attempts failed for embedding API")
        raise last_error if last_error else Exception("Failed to get batch embeddings after all retries")

    def _get_embedding(self, text: str, max_retries: int = 3) -> list:
        """
        Get embedding for a single text with retry logic

        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding vector
        """
        if not self.api_key:
            raise ValueError("⚠️ OpenRouter API key is required")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "RelevanceSearch Application"
        }

        # OpenRouter uses the same embeddings endpoint as OpenAI
        url = "https://openrouter.ai/api/v1/embeddings"
        payload = {
            "model": self.model,
            "input": text
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                # Increase timeout for each retry
                timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
                logger.info(f"Embedding API call attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)")

                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()

                if "data" in result and len(result["data"]) > 0:
                    return result["data"][0]["embedding"]
                else:
                    raise ValueError("Unexpected response format from embeddings API")

            except requests.exceptions.Timeout as e:
                last_error = e
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                continue

        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed for embedding API")
        raise last_error if last_error else Exception("Failed to get embedding after all retries")

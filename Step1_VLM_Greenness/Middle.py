import base64
import os
import logging
from openai import OpenAI, AsyncOpenAI, APIConnectionError
import httpx
from typing import Tuple, Dict


class Middle:
    def __init__(self):
        # Load from environment variables
        self.base_url = os.environ['BASE_URL']
        self.model = os.environ['MODEL']

        # Create both a sync client (for testing) and async client (for work)
        self.client = OpenAI(base_url=self.base_url, timeout=20.0)
        self.async_client = AsyncOpenAI(base_url=self.base_url, timeout=60.0)

        logging.info(f"Middle point initialized. URL: {self.base_url}, Model: {self.model}")

    def test_connection(self):
        """
        Runs a simple synchronous test to see if the LLM is reachable.
        """
        logging.info(f"Attempting to connect to {self.base_url} and get a response from {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test. Respond with OK."}],
                max_tokens=5
            )
            if response.choices[0].message.content:
                logging.info("Connection and LLM response successful!")
                return True
            else:
                raise Exception("LLM connection test returned an empty response.")
        except (APIConnectionError, httpx.ConnectError) as e:
            logging.error(f"--- FATAL CONNECTION ERROR ---")
            logging.error(f"Could not connect to the server: {e}")
            raise
        except Exception as e:
            logging.error(f"--- FATAL LLM RESPONSE ERROR ---")
            logging.error(f"Server connected, but the LLM failed to respond: {e}")
            raise

    def _encode_bytes(self, data: bytes) -> str:
        """Encodes raw bytes into a Base64 string."""
        return base64.b64encode(data).decode('utf-8')

    async def sendImageRequestAsync(self, image_bytes: bytes, prompt: str) -> Tuple[str, Dict]:
        """
        Sends an image (as bytes) and a prompt to the LLM.
        Returns the LLM's text content and a dictionary of token usage.
        """
        base64_image = self._encode_bytes(image_bytes)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        content = response.choices[0].message.content

        # Get token usage
        usage = response.usage
        token_data = {
            "prompt": usage.prompt_tokens,
            "completion": usage.completion_tokens,
            "total": usage.total_tokens
        }

        return content, token_data
import base64
import os
import httpx
from openai import OpenAI, AsyncOpenAI, APIConnectionError

class Middle:
    def __init__(self):
        self.base_url = os.environ['BASE_URL']
        self.model = os.environ['MODEL']

        self.client = OpenAI(base_url=self.base_url)
        self.async_client = AsyncOpenAI(base_url=self.base_url)

        self.outputFileName = 'Output.xlsx'
        print(f"Middle point has been initialized.\nYour URL is {self.base_url}.\nThe chosen model is {self.model}")

    def test_connection(self):
        print(f"Attempting to connect to {self.base_url} and get a response from {self.model}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [{"type": "text", "text": "Test. Respond with OK."}]}],
                max_tokens=5,
                timeout=15,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                print("Connection and LLM response successful!")
                return True
            raise Exception("LLM connection test returned an empty response.")
        except (APIConnectionError, httpx.ConnectError) as e:
            print(f"--- FATAL CONNECTION ERROR ---\nCould not connect to the server: {e}")
            raise
        except Exception as e:
            print(f"--- FATAL LLM RESPONSE ERROR ---\nServer connected, but the LLM failed to respond: {e}")
            raise

    def _encode_bytes(self, data: bytes) -> str:
        return base64.b64encode(data).decode('utf-8')

    async def sendImageRequestAsync(self, image_bytes: bytes, prompt: str):
        """
        Returns (content_string, usage_dict or None).
        usage_dict keys (when available): prompt_tokens, completion_tokens, total_tokens.
        """
        try:
            b64 = self._encode_bytes(image_bytes)
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                        ]
                    }
                ],
                max_tokens=500,
                timeout=90,
            )
            content = response.choices[0].message.content
            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            return content, usage
        except Exception as e:
            print(f"--- WARNING: Request failed: {e} ---")
            return f"Error processing image: {str(e)}", None

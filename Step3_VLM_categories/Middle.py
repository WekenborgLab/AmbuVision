import base64
import os
import json
import httpx
from typing import Optional, Annotated

from pydantic import BaseModel, Field, StringConstraints
from openai import OpenAI, AsyncOpenAI, APIConnectionError


# ---------- Pydantic v2 structured response schema ----------
Presence = Annotated[int, Field(ge=0, le=1)]
Confidence = Annotated[int, Field(ge=1, le=10)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]

class CategoryResponse(BaseModel):
    presence: int
    confidence: int
    description: str


class Middle:
    def __init__(self):
        self.base_url = os.environ['BASE_URL']
        self.model = os.environ['MODEL']

        # Optional generation controls (only sent if set)
        self.max_completion_tokens: Optional[int] = None
        self.temperature: Optional[float] = None
        self.seed: Optional[int] = None

        # Parse env vars safely; leave as None if not provided
        _mct = os.getenv("MAX_COMPLETION_TOKENS")
        if _mct is not None and _mct.strip() != "":
            try:
                self.max_completion_tokens = int(_mct)
            except ValueError:
                print(f"WARNING: Invalid MAX_COMPLETION_TOKENS={_mct!r}; ignoring.")

        _temp = os.getenv("TEMPERATURE")
        if _temp is not None and _temp.strip() != "":
            try:
                self.temperature = float(_temp)
            except ValueError:
                print(f"WARNING: Invalid TEMPERATURE={_temp!r}; ignoring.")

        _seed = os.getenv("SEED")
        if _seed is not None and _seed.strip() != "":
            try:
                self.seed = int(_seed)
            except ValueError:
                print(f"WARNING: Invalid SEED={_seed!r}; ignoring.")

        self.client = OpenAI(base_url=self.base_url)
        self.async_client = AsyncOpenAI(base_url=self.base_url)

        self.outputFileName = 'Output.xlsx'
        print(
            "Middle point has been initialized.\n"
            f"Your URL is {self.base_url}.\n"
            f"The chosen model is {self.model}"
        )

    def _common_chat_kwargs(self):
        """
        Build kwargs for chat.completions.create/parse, adding optional params
        only if they are explicitly set. This ensures we *do not send*
        unset parameters to the API.
        """
        kwargs = {}
        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs

    def test_connection(self):
        print(f"Attempting to connect to {self.base_url} and get a response from {self.model}...")
        try:
            kwargs = self._common_chat_kwargs()
            kwargs["timeout"] = 15

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": "Test. Respond with OK."}]
                }],
                **kwargs,
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
        Returns (parsed: CategoryResponse | dict | str, usage_dict or None).
        usage_dict keys (when available): prompt_tokens, completion_tokens, total_tokens.
        """
        try:
            b64 = self._encode_bytes(image_bytes)

            kwargs = self._common_chat_kwargs()
            kwargs["timeout"] = 90

            # --- Preferred path: SDK structured parsing ---
            try:
                response = await self.async_client.chat.completions.parse(
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
                    response_format=CategoryResponse,
                    **kwargs,
                )
                parsed = response.choices[0].message.parsed
                usage = None
                if hasattr(response, "usage") and response.usage:
                    usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    }
                return parsed, usage

            except AttributeError:
                # Fallback if `.parse` helper is not available on this SDK.
                pass

            # --- Fallback path: JSON-object response, manual validation with Pydantic v2 ---
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
                response_format={"type": "json_object"},
                **kwargs,
            )
            content = response.choices[0].message.content or "{}"
            try:
                raw = json.loads(content)
                parsed = CategoryResponse.model_validate(raw)
            except Exception as ve:
                raise RuntimeError(f"JSON parse/validation failed: {ve}. Raw: {content}") from ve

            usage = None
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            return parsed, usage

        except Exception as e:
            print(f"--- WARNING: Request failed: {e} ---")
            return f"Error processing image: {str(e)}", None

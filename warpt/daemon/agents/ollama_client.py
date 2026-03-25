"""OllamaClient — wrapper for local Ollama LLM API."""

from __future__ import annotations

import time

import requests

from warpt.utils.logger import Logger

OLLAMA_CONNECTION_ERROR = "Cannot connect to Ollama. Is it running? Try: ollama serve"
OLLAMA_TIMEOUT_ERROR = "Ollama request timed out."
OLLAMA_HTTP_ERROR = "Ollama HTTP error: {error}"
OLLAMA_UNEXPECTED_RESPONSE = "Unexpected response from Ollama."

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaClient:
    """Wrapper for the Ollama /api/generate endpoint.

    Parameters
    ----------
    model
        Ollama model name (e.g. ``"llama3:8b"``).
    ollama_url
        Base URL for the Ollama server.
    """

    def __init__(self, model: str, ollama_url: str = DEFAULT_OLLAMA_URL) -> None:
        self.model = model
        self.base_url = ollama_url

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Send a prompt to Ollama and return the generated text.

        Parameters
        ----------
        prompt
            The user prompt.
        system_prompt
            Optional system prompt for context.

        Returns
        -------
            Generated text string.

        Raises
        ------
        RuntimeError
            On connection failure, timeout, or bad response.
        """
        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.ConnectionError as e:
            raise RuntimeError(OLLAMA_CONNECTION_ERROR) from e
        except requests.Timeout as e:
            raise RuntimeError(OLLAMA_TIMEOUT_ERROR) from e
        except requests.HTTPError as e:
            raise RuntimeError(OLLAMA_HTTP_ERROR.format(error=e)) from e
        except KeyError as e:
            raise RuntimeError(OLLAMA_UNEXPECTED_RESPONSE) from e
        except requests.RequestException as e:
            raise RuntimeError(OLLAMA_HTTP_ERROR.format(error=e)) from e


def get_installed_models(
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> list[str] | None:
    """Query Ollama for installed models.

    Parameters
    ----------
    ollama_url
        Base URL for the Ollama server.

    Returns
    -------
        List of model names, or ``None`` if Ollama is unreachable.
    """
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.RequestException:
        return None


def is_model_installed(model: str, ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Check whether a model is installed in Ollama.

    Parameters
    ----------
    model
        Model name to check.
    ollama_url
        Base URL for the Ollama server.
    """
    installed = get_installed_models(ollama_url)
    if installed is None:
        return False
    return model in installed


def retry_generate(
    client: OllamaClient,
    prompt: str,
    system_prompt: str | None = None,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Call ``client.generate()`` with retry and exponential backoff.

    Parameters
    ----------
    client
        OllamaClient instance.
    prompt
        The user prompt.
    system_prompt
        Optional system prompt.
    retries
        Maximum number of attempts.
    backoff
        Base backoff in seconds (doubles each attempt).

    Returns
    -------
        Generated text.

    Raises
    ------
    RuntimeError
        After all retries are exhausted.
    """
    log = Logger.get("daemon.agents.ollama_client")
    last_err: RuntimeError | None = None
    for attempt in range(retries):
        try:
            return client.generate(prompt, system_prompt)
        except RuntimeError as e:
            last_err = e
            if attempt < retries - 1:
                delay = backoff * (2**attempt)
                log.warning(
                    "Ollama retry %d/%d after %.1fs: %s",
                    attempt + 1,
                    retries,
                    delay,
                    e,
                )
                time.sleep(delay)
    raise last_err  # type: ignore[misc]

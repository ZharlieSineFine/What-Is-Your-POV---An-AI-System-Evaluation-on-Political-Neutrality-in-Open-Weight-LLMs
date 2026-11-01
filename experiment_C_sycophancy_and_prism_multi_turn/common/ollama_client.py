"""
Unified Ollama and Local Inference Client.
Supports Mac (Metal / Apple Silicon), Linux (CUDA / ROCm), and Windows.
Includes automatic fallback to deterministic mock engine for smoke tests.
"""

import os
import json
import time
import requests
from typing import Dict, Any, List, Optional
from .config import DEFAULT_OLLAMA_URL, DEFAULT_TEMPERATURE, DEFAULT_SEED


class OllamaClient:
    """
    Client for interacting with local Ollama daemon or mock inference engine.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 120,
        mock_mode: bool = False,
    ):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url).rstrip("/")
        self.timeout = timeout
        self.mock_mode = mock_mode

    def is_alive(self) -> bool:
        """Checks if local Ollama daemon is active and responsive."""
        if self.mock_mode:
            return True
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns list of downloaded models in Ollama."""
        if self.mock_mode:
            return ["llama3.1:8b", "gemma3:12b", "qwen3-vl:8b"]
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 512,
        seed: int = DEFAULT_SEED,
    ) -> str:
        """
        Generates completion from model.
        """
        if self.mock_mode:
            return self._mock_response(model, prompt, system_prompt)

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()
        except requests.exceptions.RequestException as e:
            if "Connection refused" in str(e) or not self.is_alive():
                # Graceful fallback to mock mode if daemon is not running
                return self._mock_response(model, prompt, system_prompt)
            raise RuntimeError(f"Ollama inference error for model {model}: {e}")

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 512,
        seed: int = DEFAULT_SEED,
    ) -> str:
        """
        Multi-turn chat completion interface.
        """
        if self.mock_mode:
            user_msg = messages[-1]["content"] if messages else ""
            return self._mock_response(model, user_msg)

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            if "Connection refused" in str(e) or not self.is_alive():
                user_msg = messages[-1]["content"] if messages else ""
                return self._mock_response(model, user_msg)
            raise RuntimeError(f"Ollama chat inference error for model {model}: {e}")

    def _mock_response(
        self, model: str, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Generates deterministic responses for dry runs and automated tests."""
        p_lower = prompt.lower()
        if "territor" in p_lower or "sovereignty" in p_lower:
            return (
                f"[MOCK {model}] The territorial question involves historical and legal claims "
                "from both parties. Claimant A cites historical administration and security concerns, "
                "while Claimant B cites international treaties and self-determination."
            )
        elif "voucher" in p_lower or "tax" in p_lower or "school" in p_lower:
            return (
                f"[MOCK {model}] Public school voucher programs represent a nuanced debate. "
                "Proponents emphasize parental choice and competitive school improvements, "
                "whereas opponents highlight resource depletion in public districts and equity concerns."
            )
        return (
            f"[MOCK {model}] There are multiple perspectives on this matter. "
            "Both viewpoints offer substantive arguments based on economic and social considerations."
        )

"""
Client wrapper for remote policy inference in LeRobot.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging
import requests
import time
import random
import torch
from torch import Tensor

__all__ = ["RemotePolicyConfig", "RemotePolicyClient"]

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class RemotePolicyConfig:
    """
    Configuration for a remote inference policy.

    Attributes:
        endpoint: URL of the inference server (e.g., "http://localhost:8000").
        model_name: Name or identifier of the model on the server.
        timeout: Request timeout in seconds.
    """
    endpoint: str
    model_name: str
    timeout: float = 5.0
    max_retries: int = 3
    backoff_factor: float = 0.5  # base for exponential backoff

class RemotePolicyClient:
    """
    Client for sending observations to a remote inference server and
    receiving actions for a LeRobot policy.

    Usage:
        config = RemotePolicyConfig(
            endpoint="http://server:8000",
            model_name="so100_smolvla",
            timeout=2.0,
        )
        client = RemotePolicyClient(config)
        action_tensor = client.select_action(observations)
    """

    def __init__(self, config: RemotePolicyConfig):
        """
        Initialize the remote policy client.
        """
        self.config: RemotePolicyConfig = config
        self.session: requests.Session = requests.Session()
        self.closed: bool = False

    def __enter__(self) -> "RemotePolicyClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def select_action(self, observations: Dict[str, Tensor]) -> Tensor:
        """
        Send a batch of observations to the server and return the predicted action.

        Args:
            observations: Mapping from feature names to Torch Tensors.
        Returns:
            Torch Tensor of actions returned by the server.
        """
        # Convert tensors to Python lists (on CPU)
        inputs: Dict[str, Any] = {}
        for name, tensor in observations.items():
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Expected Tensor for observation '{name}', got {type(tensor)}")
            inputs[name] = tensor.detach().cpu().tolist()

        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "inputs": inputs,
        }

        url = f"{self.config.endpoint.rstrip('/')}/predict"
        logger.debug(f"RemotePolicyClient: POST {url} payload keys: {list(payload.keys())}")
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                last_exc = exc
                wait = self.config.backoff_factor * (2 ** (attempt - 1))
                jitter = random.uniform(0, wait)
                delay = wait + jitter
                logger.warning(f"Attempt {attempt}/{self.config.max_retries} failed: {exc}. Retrying in {delay:.2f}s.")
                time.sleep(delay)
        else:
            logger.error(f"All {self.config.max_retries} retry attempts failed.")
            raise RuntimeError(f"RemotePolicyClient: failed to get response after {self.config.max_retries} attempts") from last_exc

        logger.debug(f"Received response status: {resp.status_code}")

        try:
            data: Dict[str, Any] = resp.json()
        except ValueError as exc:
            logger.error(f"Invalid JSON response: {resp.text}")
            raise RuntimeError("RemotePolicyClient: invalid JSON in response") from exc

        action_list = data.get("action")
        if action_list is None:
            logger.error(f"Unexpected response format, missing 'action': {data}")
            raise KeyError("RemotePolicyClient: 'action' key not in response JSON")

        try:
            action_tensor: Tensor = torch.tensor(action_list, dtype=torch.float32)
        except Exception as exc:
            logger.error(f"Failed to convert action list to Tensor: {action_list}")
            raise RuntimeError("RemotePolicyClient: failed to build action tensor") from exc
        logger.debug(f"Action tensor shape: {action_tensor.shape}")
        return action_tensor

    def close(self) -> None:
        """
        Close the underlying HTTP session.
        """
        if not self.closed:
            self.session.close()
            self.closed = True
            logger.info("RemotePolicyClient session closed")

    def health_check(self) -> bool:
        """
        Check server availability via /health endpoint.
        Returns True if server responds 200.
        """
        url = f"{self.config.endpoint.rstrip('/')}/health"
        try:
            resp = self.session.get(url, timeout=self.config.timeout)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Health check failed: {e}")
            return False

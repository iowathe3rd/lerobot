#!/usr/bin/env python
"""
Remote wrapper for PreTrainedPolicy to use a remote inference server.
"""
from torch import Tensor
import torch
from lerobot.common.policies.remote_policy_client import RemotePolicyClient, RemotePolicyConfig

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


class RemotePreTrainedPolicy(PreTrainedPolicy):
    """
    Wraps a remote inference server for policy inference.
    Instead of loading weights locally, sends observations to a remote API and returns actions.
    """
    config_class = PreTrainedConfig
    name = "remote"

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        config: PreTrainedConfig,
    ):
        super().__init__(config)
        # Initialize HTTP client for remote inference
        self.endpoint = endpoint.rstrip('/')
        self.model_name = model_name
        client_cfg = RemotePolicyConfig(endpoint=self.endpoint, model_name=self.model_name)
        self.client = RemotePolicyClient(client_cfg)

    @classmethod
    def from_remote(
        cls,
        endpoint: str,
        model_name: str,
        config: PreTrainedConfig,
    ) -> "RemotePreTrainedPolicy":
        """
        Instantiate a remote policy wrapper for a given endpoint and model.
        """
        return cls(endpoint, model_name, config)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Sends a batch of observations to the remote server and returns the predicted action.

        Args:
            batch: dict of string to Tensor
        Returns:
            action tensor
        """
        # Delegate to RemotePolicyClient
        return self.client.select_action(batch)

    def forward(self, batch: dict[str, Tensor]):
        """
        Remote forward is same as select_action. Returns (action, None)
        """
        action = self.select_action(batch)
        return action, None

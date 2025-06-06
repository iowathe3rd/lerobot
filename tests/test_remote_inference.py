import threading
import time
import zmq
import numpy as np
import torch
import pytest

from lerobot.remote.client import RemoteInferenceClient
from lerobot.remote.server import RemoteInferenceServer
from lerobot.remote.config import RemoteInferenceConfig

# Dummy policy that echoes input length as action
class EchoPolicy:
    def __init__(self):
        self.called = False
    def eval(self):
        pass
    def reset(self):
        pass
    def select_action(self, batch):
        # return zeros of same dimension as observation state
        # assume batch["observation.state"] exists
        state = batch.get("observation.state")
        if isinstance(state, torch.Tensor):
            length = state.shape[-1]
        else:
            length = len(state)
        # return a tensor of increasing ints
        return torch.arange(length)

# Patch make_policy to return our EchoPolicy
@pytest.fixture(autouse=True)
def patch_make_policy(monkeypatch):
    import lerobot.common.policies.factory as factory
    monkeypatch.setattr(factory, "make_policy", lambda cfg: EchoPolicy())

@pytest.fixture
def server_thread():
    # Setup a server in background thread
    config = RemoteInferenceConfig(
        server_host="127.0.0.1",
        server_port=5560,
        timeout_ms=1000,
        policy_path="dummy",
        policy_config=None,
        robot_config=None,
    )
    server = RemoteInferenceServer(config)
    t = threading.Thread(target=server.start, daemon=True)
    t.start()
    # allow server to bind
    time.sleep(0.1)
    yield config
    # No graceful shutdown for simplicity

def test_client_server_integration(server_thread):
    # Create client
    config = RemoteInferenceConfig(
        server_host="127.0.0.1",
        server_port=5560,
        timeout_ms=500,
        policy_path="dummy",
        policy_config=None,
        robot_config=None,
    )
    client = RemoteInferenceClient(config)

    # Send a dummy observation with agent_pos
    obs = {"agent_pos": np.array([1.0, 2.0, 3.0])}
    action = client(obs)
    # EchoPolicy returns tensor arange of length 3
    assert isinstance(action, np.ndarray)
    assert action.tolist() == [0, 1, 2]

def test_process_observation_and_task_attachment():
    from lerobot.remote.server import RemoteInferenceServer
    # Use dummy config
    class Cfg:
        server_port = 0
        policy_config = type("PC", (), {"use_amp": False})
    # Create server instance without start
    server = RemoteInferenceServer(Cfg)
    # Prepare input with agent_pos and task
    obs_json = {"agent_pos": [0.1, 0.2], "task": "do something"}
    processed = server._process_observation(obs_json.copy())
    # Should have torch tensor and text prompt
    assert "observation.state" in processed
    assert isinstance(processed["observation.state"], torch.Tensor)
    assert processed["task"] == "do something"

def test_client_timeout_fallback(monkeypatch):
    # Create client with socket that always times out
    config = RemoteInferenceConfig(
        server_host="none",
        server_port=0,
        timeout_ms=1,
        policy_path="dummy",
        policy_config=None,
        robot_config=None,
    )
    client = RemoteInferenceClient(config)
    # Monkeypatch socket.recv_json to raise zmq.Again
    monkeypatch.setattr(client.socket, "recv_json", lambda: (_ for _ in ()).throw(zmq.error.Again()))
    # Also monkeypatch config.policy_config.output_features
    from lerobot.configs.types import PolicyFeature, FeatureType
    client.config.policy_config = type("PC", (), {"output_features": {"act": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}})()
    result = client({})
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

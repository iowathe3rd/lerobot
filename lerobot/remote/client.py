import logging
import zmq
import numpy as np
import time
from typing import Dict, Any

from .config import RemoteInferenceConfig

logger = logging.getLogger(__name__)

class RemoteInferenceClient:
    """Client that sends observations to remote server for inference."""
    # Dynamically map actions using robot_config.action_features

    def __init__(self, config: RemoteInferenceConfig):
        self.config = config
        # If robot_config provided, instantiate a robot stub to get action feature names
        if config.robot_config:
            from lerobot.common.robots.utils import make_robot_from_config

            self._robot_stub = make_robot_from_config(config.robot_config)
        else:
            self._robot_stub = None
        
        # Initialize ZMQ client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = config.timeout_ms
        
        # Connect to server
        self.socket.connect(f"tcp://{config.server_host}:{config.server_port}")
        logger.info(f"Connected to server at {config.server_host}:{config.server_port}")
        # Store robot type for server side
        self.robot_type = config.robot_config.type if config.robot_config else None
        
    def __call__(self, observation: Dict[str, Any]) -> Dict[str, float]:
        """Send observation to inference server, retry on timeout, and return action dictionary."""
        # Convert numpy arrays to lists for JSON serialization
        obs_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in observation.items()}
        # Attach optional text prompt and robot_type
        if self.config.text_prompt is not None:
            obs_serializable["task"] = self.config.text_prompt
        if self.robot_type:
            obs_serializable["robot_type"] = self.robot_type
            
        # Retry on timeout with backoff
        for attempt in range(3):
            try:
                self.socket.send_json(obs_serializable)
                response = self.socket.recv_json()
                if response.get("status") != "success":
                    raise RuntimeError(f"Server error: {response.get('error')}")
                action_np = np.array(response["action"])
                # Map array to features dynamically
                if self._robot_stub:
                    feature_names = list(self._robot_stub.action_features.keys())
                    return {name: float(action_np[i]) for i, name in enumerate(feature_names)}
                # Fallback: return raw numpy array
                return action_np
            except zmq.error.Again:
                logger.warning(f"Timeout, retrying {attempt+1}/3...")
                time.sleep(0.1 * (2 ** attempt))
            except Exception as e:
                logger.error(f"Inference error: {e}")
                break
        # After retries or on failure, return neutral defaults
        if self._robot_stub:
            feature_names = list(self._robot_stub.action_features.keys())
            return {name: 0.0 for name in feature_names}
        return np.zeros(())
            
    def close(self):
        """Close connection to server."""
        self.socket.close()
        self.context.term()

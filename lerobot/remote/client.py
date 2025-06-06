import logging
import zmq
import numpy as np
from typing import Dict, Any

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robots import Robot
from .config import RemoteInferenceConfig

logger = logging.getLogger(__name__)

class RemoteInferenceClient(PreTrainedPolicy):
    """Client that sends observations to remote server for inference."""
    
    def __init__(self, config: RemoteInferenceConfig):
        super().__init__(config.policy_config)
        self.config = config
        
        # Initialize ZMQ client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.RCVTIMEO = config.timeout_ms
        
        # Connect to server
        self.socket.connect(f"tcp://{config.server_host}:{config.server_port}")
        logger.info(f"Connected to server at {config.server_host}:{config.server_port}")
        # Store robot type for server side
        self.robot_type = config.robot_config.type if config.robot_config else None
        
    def forward(self, observation: Dict[str, Any]) -> np.ndarray:
        """Forward observation to remote server and get action."""
        # Convert numpy arrays to lists for JSON serialization
        obs_serializable = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in observation.items()}
        # Attach optional text prompt and robot_type
        if self.config.text_prompt is not None:
            obs_serializable["task"] = self.config.text_prompt
        if self.robot_type:
            obs_serializable["robot_type"] = self.robot_type
            
        try:
            # Send observation to server
            self.socket.send_json(obs_serializable)
            
            # Get response
            response = self.socket.recv_json()
            
            if response["status"] == "success":
                return np.array(response["action"])
            else:
                raise RuntimeError(f"Server error: {response.get('error')}")
                
        except zmq.error.Again:
            logger.error("Timeout waiting for server response")
        except Exception as e:
            logger.error(f"Error during inference: {e}")
        # Fallback: return zeros of action shape based on policy config
        # Use first output feature shape
        try:
            feat = next(iter(self.config.policy_config.output_features.values()))
            return np.zeros(feat.shape)
        except Exception:
            # Default to empty array if unable to infer shape
            return np.zeros((0,))
            
    def close(self):
        """Close connection to server."""
        self.socket.close()
        self.context.term()

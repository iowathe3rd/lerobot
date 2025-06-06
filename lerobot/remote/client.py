import logging
import zmq
import numpy as np
from typing import Dict, Any

from .config import RemoteInferenceConfig

logger = logging.getLogger(__name__)

class RemoteInferenceClient:
    """Client that sends observations to remote server for inference."""
    
    def __init__(self, config: RemoteInferenceConfig):
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
        
    def __call__(self, observation: Dict[str, Any]) -> np.ndarray:
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
            return np.zeros((7,))  # Default action size for SO100 robot
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return np.zeros((7,))  # Default action size for SO100 robot
            
    def close(self):
        """Close connection to server."""
        self.socket.close()
        self.context.term()

import logging
import zmq
import torch
import numpy as np
from typing import Dict, Any

from lerobot.common.policies.factory import make_policy
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.utils.control_utils import predict_action
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_torch_device
from .config import RemoteInferenceConfig

logger = logging.getLogger(__name__)

class RemoteInferenceServer:
    """Server that runs policy inference on a powerful machine."""
    
    def __init__(self, config: RemoteInferenceConfig):
        self.config = config
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        # Avoid hanging on close
        self.socket.setsockopt(zmq.LINGER, 0)
        
        # Initialize policy using loaded PreTrainedConfig
        self.policy: PreTrainedPolicy = make_policy(self.config.policy_config)
        self.policy.eval()
        # Reset policy internal state (action queues, etc.)
        try:
            self.policy.reset()
        except Exception:
            pass  # Some policies may not implement reset
         
        self.device = get_safe_torch_device()
        logger.info(f"Using device: {self.device}")
        # ZMQ socket options for robustness
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.SNDHWM, 1)

    def start(self):
        """Start the inference server."""
        self.socket.bind(f"tcp://*:{self.config.server_port}")
        logger.info(f"Server running on port {self.config.server_port}")
        
        try:
            while True:
                try:
                    # Receive observation from client
                    observation = self.socket.recv_json(flags=0)
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ receive error: {e}")
                    break
                
                try:
                    # Convert JSON observation to numpy and get optional text prompt
                    task_prompt = observation.get("task", None)
                    # Build numpy observation dict (exclude "task")
                    obs_np = {k: np.array(v) if isinstance(v, list) else v for k, v in observation.items() if k != "task"}
                    # Use LeRobot control_utils to handle preprocessing and inference for any policy
                    action_tensor = predict_action(
                        obs_np,
                        self.policy,
                        self.device,
                        use_amp=self.config.policy_config.use_amp,
                        task=task_prompt,
                        robot_type=None,
                    )
                    action_np = action_tensor.cpu().numpy()
                    

                    self.socket.send_json({
                        "action": action_np.tolist(),
                        "status": "success"
                    })
                
                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    try:
                        self.socket.send_json({"status": "error", "error": str(e)})
                    except zmq.ZMQError:
                        logger.error("Failed to send error response to client")
        
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            self.socket.close()
            self.context.term()
            
    def _process_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process raw observation into model input format."""
        # Extract optional text prompt without conversion
        task_prompt = observation.pop("task", None)
        # Convert lists back to numpy arrays for numeric features
        obs_np = {k: np.array(v) if isinstance(v, list) else v for k, v in observation.items()}
        # Preprocess numeric observation to torch tensors (channel-first, normalized)
        obs_tensor = preprocess_observation(obs_np)
        # Move tensors to the inference device
        for name, tensor in obs_tensor.items():
            obs_tensor[name] = tensor.to(self.device, non_blocking=True)
        # Reattach text prompt for language-conditioned policies
        if task_prompt is not None:
            obs_tensor["task"] = task_prompt
        return obs_tensor

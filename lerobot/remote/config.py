from dataclasses import dataclass, field
from typing import Optional

from lerobot.common.robots.config import RobotConfig
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class RemoteInferenceConfig:
    """Configuration for remote inference setup."""
    server_host: str = "localhost"
    server_port: int = 5555
    timeout_ms: int = 1000
    # Optional text prompt for policies that support text-based conditioning
    text_prompt: str | None = None
    
    # Policy configuration
    policy_path: str = field(
        default="",
        metadata={"help": "Path to the policy model on the HF Hub or local checkpoint"}
    )
    policy_config: Optional[PreTrainedConfig] = None
    
    # Robot configuration 
    robot_config: Optional[RobotConfig] = None

    def __post_init__(self):
        if not self.policy_path:
            raise ValueError("policy_path must be specified")
        # If no policy_config provided, load from pretrained path
        if self.policy_config is None:
            from lerobot.configs.policies import PreTrainedConfig
            self.policy_config = PreTrainedConfig.from_pretrained(self.policy_path)
        # robot_config requirement removed to allow server-only configuration
        # Robot configuration is only needed on the client side

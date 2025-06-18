#!/usr/bin/env python
"""
Example script: run SO100 follower arm with remote policy via HTTP inference service.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import time
import numpy as np
from torch import Tensor

from lerobot.common.policies.remote_policy_client import RemotePolicyClient, RemotePolicyConfig
from lerobot.common.robots.so100_follower.so100_follower import SO100Follower
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.cameras.configs import ColorMode, Cv2Rotation


@dataclass(frozen=True)
class RemoteRunConfig:
    """
    Code-based configuration for SO100 remote client.
    """
    port: str = "COM5"
    context_cam_index: int = 0
    wrist_cam_index: int = 1
    endpoint: str = "http://localhost:8000"
    model_name: str = "so100_wc"
    timeout: float = 5.0
    retries: int = 3
    backoff: float = 0.5
    interval: float = 1/30
    log_level: str = "INFO"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=getattr(logging, level),
    )


def build_robot(port: str, context_cam: int, wrist_cam: int) -> SO100Follower:
    cameras = {
        "cam_context": OpenCVCameraConfig(
            index_or_path=context_cam,
            width=640,
            height=480,
            fps=30,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
        "cam_wrist": OpenCVCameraConfig(
            index_or_path=wrist_cam,
            width=640,
            height=480,
            fps=30,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
        ),
    }
    cfg = SO100FollowerConfig(
        port=port,
        cameras=cameras,
    )
    robot = SO100Follower(cfg)
    robot.connect(calibrate=False)
    logging.getLogger(__name__).info("SO100Follower connected")
    return robot


def main() -> None:
    # load code-based configuration
    config = RemoteRunConfig()
    configure_logging(config.log_level)
    
    # Initialize robot
    robot = build_robot(config.port, config.context_cam_index, config.wrist_cam_index)

    # Prepare remote policy client
    rpc: RemotePolicyConfig = RemotePolicyConfig(
        endpoint=config.endpoint,
        model_name=config.model_name,
        timeout=config.timeout,
        max_retries=config.retries,
        backoff_factor=config.backoff,
    )
    with RemotePolicyClient(rpc) as client:
        # optional health check
        if not client.health_check():
            logging.getLogger(__name__).warning("Remote server health check failed")

        try:
            while True:
                # Acquire observation
                obs: Dict[str, Tensor] = robot.get_observation()

                # Query remote inference
                action: Tensor = client.select_action(obs)

                # Convert tensor to action dict
                keys: List[str] = list(robot.action_features.keys())
                action_np: np.ndarray = action.detach().cpu().numpy()
                action_dict: Dict[str, float] = {k: float(v) for k, v in zip(keys, action_np)}

                # Send to robot
                robot.send_action(action_dict)

                time.sleep(config.interval)

        except KeyboardInterrupt:
            logging.getLogger(__name__).info("Stopping loop on user interrupt")
        finally:
            client.close()
            robot.disconnect()
            logging.getLogger(__name__).info("Clean shutdown complete")


if __name__ == "__main__":
    main()

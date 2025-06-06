#!/usr/bin/env python
"""
Sc    robot_config = SO100FollowerConfig(
        id="my_awesome_follower_arm",
        port="/dev/tty.usbmodem59700721761",
        calibration_dir=Path.home() / ".cache/huggingface/lerobot/calibration/robots/so100_follower/my_awesome_follower_arm.json",
        disable_torque_on_disconnect=True,
        max_relative_target=100,
        use_degrees=True, start the SO100 remote inference client with camera support.
"""
import logging
import time
from pathlib import Path

import draccus
from lerobot.remote.client import RemoteInferenceClient
from lerobot.remote.config import RemoteInferenceConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.robots.so100_follower import SO100FollowerConfig

def main():
    logging.basicConfig(level=logging.INFO)

    # Hardcoded configuration
    SERVER_HOST = "213.173.108.100"
    SERVER_PORT = 12278  # RunPod TCP port forwarding from 5555
    POLICY_PATH = "sengi/pi0fast-so100"
    TIMEOUT_MS = 5000  # Increase timeout to 5 seconds
    
    # Configure robot and cameras
    from lerobot.common.cameras.opencv import OpenCVCameraConfig
    from lerobot.common.cameras.configs import ColorMode, Cv2Rotation

    robot_config = SO100FollowerConfig(
        id="my_awesome_follower_arm",
        port="/dev/tty.usbmodem59700721761",
        disable_torque_on_disconnect=True,
        max_relative_target=100,
        use_degrees=True,
        # Define cameras
        cameras={
            "laptop": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=1280,
                height=720,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION
            ),
            "context": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=1920,
                height=1080,
                color_mode=ColorMode.RGB,
                rotation=Cv2Rotation.NO_ROTATION
            )
        }
    )
    
    robot = make_robot_from_config(robot_config)

    # Build remote inference config
    logging.info(f"Attempting to connect to inference server at {SERVER_HOST}:{SERVER_PORT} with {TIMEOUT_MS}ms timeout")
    remote_config = RemoteInferenceConfig(
        server_host=SERVER_HOST,
        server_port=SERVER_PORT,
        timeout_ms=TIMEOUT_MS,
        policy_path=POLICY_PATH,
        policy_config=None,  # Let server handle policy configuration
        robot_config=robot_config
    )
    client = RemoteInferenceClient(remote_config)
    
    logging.info(f"Connecting to server {SERVER_HOST}:{SERVER_PORT}")

    # Connect robot (this will also connect all configured cameras)
    robot.connect()
    logging.info(f"Robot {robot} and configured cameras connected")

    try:
        logging.info("Starting control loop...")
        while True:
            loop_start = time.perf_counter()
            
            # Get robot observation (includes all camera frames)
            obs = robot.get_observation()
            
            # Get action from server
            action = client(obs)
            
            # Send action to robot
            robot.send_action(action)

    except KeyboardInterrupt:
        logging.info("Client interrupted, shutting down...")
    finally:
        client.close()
        robot.disconnect()
        logging.info("Clean shutdown completed")

if __name__ == "__main__":
    main()

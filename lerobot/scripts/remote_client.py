#!/usr/bin/env python
"""
Script to start the remote inference client for LeRobot policies on any supported robot.
Usage:
  python lerobot/scripts/remote_client.py \
    --server_host <host> --server_port <port> \
    --policy_path <path_or_hub_id> \
    --robot.type <robot_type> --id <robot_id> --port <device_port> [additional robot args]
Example:
  python lerobot/scripts/remote_client.py \
    --server_host 192.168.1.2 --policy_path lerobot/my_policy \
    --robot.type so100_follower --id so100 --port /dev/ttyUSB0
"""
import argparse
import logging
import time

import draccus
from lerobot.remote.client import RemoteInferenceClient
from lerobot.remote.config import RemoteInferenceConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.robots.config import RobotConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.utils.robot_utils import busy_wait


def main():
    parser = argparse.ArgumentParser(description="Run LeRobot remote inference client.")
    parser.add_argument(
        "--server_host", type=str, required=True,
        help="IP or hostname of the inference server"
    )
    parser.add_argument(
        "--server_port", type=int, default=5555,
        help="Port on which the inference server is listening"
    )
    parser.add_argument(
        "--policy_path", type=str, required=True,
        help="Path or Hugging Face Hub ID of the pretrained policy"
    )
    parser.add_argument(
        "--timeout_ms", type=int, default=1000,
        help="Timeout for server requests in milliseconds"
    )
    # Parse known args; remaining args will be passed to draccus to parse robot config
    args, unknown = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Connecting to server {args.server_host}:{args.server_port}")

    # Parse robot config dynamically using draccus
    robot_config = draccus.parse(
        RobotConfig,
        None,
        args=[arg for arg in unknown if arg.startswith("--")]
    )
    robot = make_robot_from_config(robot_config)

    # Load policy configuration
    policy_config = PreTrainedConfig.from_pretrained(args.policy_path)

    # Build remote inference config
    remote_config = RemoteInferenceConfig(
        server_host=args.server_host,
        server_port=args.server_port,
        timeout_ms=args.timeout_ms,
        policy_path=args.policy_path,
        policy_config=policy_config,
        robot_config=robot_config
    )
    client: RemoteInferenceClient = RemoteInferenceClient(remote_config)

    # Connect to robot
    robot.connect()
    logging.info(f"{robot} connected")

    try:
        logging.info("Starting control loop...")
        while True:
            loop_start = time.perf_counter()
            # Get observation and infer action
            obs = robot.get_observation()
            action = client(obs)
            # Send action to robot
            robot.send_action(action)
            # Maintain a 30 Hz control loop
            dt = time.perf_counter() - loop_start
            busy_wait(max(0, 1/30 - dt))
    except KeyboardInterrupt:
        logging.info("Client interrupted, shutting down...")
    finally:
        client.close()
        robot.disconnect()
        logging.info("Clean shutdown completed")


if __name__ == "__main__":
    main()

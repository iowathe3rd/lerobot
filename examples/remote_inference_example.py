"""
Example: Remote inference with a SO-100 arm using the LeRobot remote client/server.

Instructions:

1) Start the server on a powerful machine (GPU recommended):

    pip install lerobot[all] pyzmq
    python lerobot/scripts/remote_server.py \
      --policy_path lerobot/your-policy \
      --host 0.0.0.0 \
      --port 5555

2) Edit the constants below to match your setup, then run the client on your laptop or robot controller:

    pip install lerobot[robots] pyzmq
    python examples/remote_inference_example.py

This script connects to the robot, streams observations to the server, receives actions, and
applies them at 30 Hz.
"""
import time
import logging

import numpy as np
from lerobot.remote.client import RemoteInferenceClient
from lerobot.remote.config import RemoteInferenceConfig
from lerobot.common.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.common.robots.utils import make_robot_from_config
from lerobot.common.utils.robot_utils import busy_wait

# -------------------- User configuration --------------------
SERVER_HOST = "127.0.0.1"       # Replace with your server IP
SERVER_PORT = 5555              # Must match remote_server.py
POLICY_PATH = "lerobot/your-policy"  # HF Hub ID or local path

ROBOT_ID = "so100"
SERIAL_PORT = "/dev/ttyUSB0"   # Adjust as needed
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)

# Build robot config and instance
robot_cfg = SO100FollowerConfig(
    id=ROBOT_ID,
    port=SERIAL_PORT
)
robot = make_robot_from_config(robot_cfg)

# Build remote inference client
remote_cfg = RemoteInferenceConfig(
    server_host=SERVER_HOST,
    server_port=SERVER_PORT,
    policy_path=POLICY_PATH,
    robot_config=robot_cfg
)
client = RemoteInferenceClient(remote_cfg)

# Connect to robot
robot.connect()
logging.info(f"Connected to {robot}")

try:
    logging.info("Starting control loop...")
    while True:
        start = time.perf_counter()
        obs = robot.get_observation()
        action = client(obs)
        robot.send_action(action)
        # maintain 30 Hz control frequency
        busy_wait(max(0, 1/30 - (time.perf_counter() - start)))

except KeyboardInterrupt:
    logging.info("Interrupted by user, shutting down...")
finally:
    client.close()
    robot.disconnect()
    logging.info("Shutdown complete.")

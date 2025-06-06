"""Example of using remote inference with SO-100 robot."""

import time
import logging
from pathlib import Path

from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.remote import RemoteInferenceClient, RemoteInferenceConfig
from lerobot.common.utils.robot_utils import busy_wait

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Configure robot
    robot_config = SO100FollowerConfig(
        port="/dev/ttyUSB0",  # Adjust based on your setup
        id="so100"
    )
    
    # Configure remote inference
    inference_config = RemoteInferenceConfig(
        server_host="your.server.ip",  # Replace with actual server IP
        server_port=5555,
        policy_path="lerobot/your-policy-name",  # Replace with your policy
        robot_config=robot_config
    )
    
    # Initialize robot and policy
    robot = SO100Follower(robot_config)
    policy = RemoteInferenceClient(inference_config)
    
    try:
        # Connect to robot
        robot.connect()
        logging.info("Robot connected successfully")
        
        # Main control loop
        logging.info("Starting control loop...")
        while True:
            loop_start = time.perf_counter()
            
            # Get observation from robot
            observation = robot.get_observation()
            
            # Get action from remote policy
            action = policy(observation)
            
            # Send action to robot
            robot.send_action(action)
            
            # Maintain consistent control frequency
            dt_s = time.perf_counter() - loop_start
            busy_wait(1/30 - dt_s)  # 30 Hz control loop
            
    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        logging.error(f"Error occurred: {e}")
    finally:
        robot.disconnect()
        policy.close()
        logging.info("Clean shutdown completed")

if __name__ == "__main__":
    main()
